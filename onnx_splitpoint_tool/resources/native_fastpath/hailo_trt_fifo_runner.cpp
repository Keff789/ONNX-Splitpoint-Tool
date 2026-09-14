#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
#include <hailo/hailort.hpp>

#ifdef HAVE_OPENCV
#include <opencv2/opencv.hpp>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

struct Logger final : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::cerr << "[TRT] " << msg << "\n";
    }
};
static Logger gLogger;

static double ms_since(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static size_t trt_dtype_size(nvinfer1::DataType t) {
    switch (t) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT8: return 1;
        case nvinfer1::DataType::kUINT8: return 1;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kBOOL: return 1;
#if NV_TENSORRT_MAJOR >= 9
        case nvinfer1::DataType::kINT64: return 8;
#endif
        default: return 4;
    }
}

static std::string trt_dtype_name(nvinfer1::DataType t) {
    switch (t) {
        case nvinfer1::DataType::kFLOAT: return "float32";
        case nvinfer1::DataType::kHALF: return "float16";
        case nvinfer1::DataType::kINT8: return "int8";
        case nvinfer1::DataType::kUINT8: return "uint8";
        case nvinfer1::DataType::kINT32: return "int32";
        case nvinfer1::DataType::kBOOL: return "bool";
#if NV_TENSORRT_MAJOR >= 9
        case nvinfer1::DataType::kINT64: return "int64";
#endif
        default: return "unknown";
    }
}

static size_t vol(nvinfer1::Dims d) {
    size_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) v *= static_cast<size_t>(std::max(1, d.d[i]));
    return v;
}

static std::string dims_to_string(nvinfer1::Dims d) {
    std::ostringstream os;
    os << "[";
    for (int i = 0; i < d.nbDims; ++i) { if (i) os << ","; os << d.d[i]; }
    os << "]";
    return os.str();
}

struct Args {
    std::string hef;
    std::string engine;
    std::string image_dir;
    std::string image;
    std::string out_json = "native_fifo_report.json";
    int warmup = 20;
    int frames = 200;
    int queue_depth = 2;
    bool copy_outputs = true;
    bool synthetic = false;
    bool verbose = false;
};

static void usage() {
    std::cerr << "splitpoint_hailo_trt_fifo --hef <part1.hef> --engine <part2.engine> [--image-dir DIR|--image IMG|--synthetic] --out report.json --warmup 20 --frames 200 --queue-depth 2\n";
}

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&](const char* name) -> std::string { if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name); return argv[++i]; };
        if (k == "--hef") a.hef = need("--hef");
        else if (k == "--engine") a.engine = need("--engine");
        else if (k == "--image-dir") a.image_dir = need("--image-dir");
        else if (k == "--image") a.image = need("--image");
        else if (k == "--out") a.out_json = need("--out");
        else if (k == "--warmup") a.warmup = std::stoi(need("--warmup"));
        else if (k == "--frames") a.frames = std::stoi(need("--frames"));
        else if (k == "--queue-depth") a.queue_depth = std::stoi(need("--queue-depth"));
        else if (k == "--no-copy-outputs") a.copy_outputs = false;
        else if (k == "--synthetic") a.synthetic = true;
        else if (k == "--verbose") a.verbose = true;
        else if (k == "--help" || k == "-h") { usage(); std::exit(0); }
        else throw std::runtime_error("unknown argument: " + k);
    }
    if (a.hef.empty() || a.engine.empty()) throw std::runtime_error("--hef and --engine are required");
    if (a.frames <= 0) throw std::runtime_error("--frames must be > 0");
    if (a.queue_depth <= 0) a.queue_depth = 1;
    return a;
}

struct BoundedQueue {
    std::mutex m;
    std::condition_variable cv_not_full, cv_not_empty;
    std::deque<std::shared_ptr<std::vector<uint8_t>>> q;
    size_t cap;
    bool closed = false;
    explicit BoundedQueue(size_t c) : cap(std::max<size_t>(1, c)) {}
    void push(std::shared_ptr<std::vector<uint8_t>> item) {
        std::unique_lock<std::mutex> lk(m);
        cv_not_full.wait(lk, [&]{ return q.size() < cap || closed; });
        if (closed) return;
        q.push_back(std::move(item));
        cv_not_empty.notify_one();
    }
    std::shared_ptr<std::vector<uint8_t>> pop() {
        std::unique_lock<std::mutex> lk(m);
        cv_not_empty.wait(lk, [&]{ return !q.empty() || closed; });
        if (q.empty()) return nullptr;
        auto v = std::move(q.front()); q.pop_front(); cv_not_full.notify_one(); return v;
    }
    void close() { std::lock_guard<std::mutex> lk(m); closed = true; cv_not_full.notify_all(); cv_not_empty.notify_all(); }
};

struct TrtBinding {
    std::string name;
    nvinfer1::TensorIOMode mode;
    nvinfer1::DataType dtype;
    nvinfer1::Dims dims;
    size_t bytes = 0;
    void* device = nullptr;
    std::vector<uint8_t> host;
};

class TrtEngine {
public:
    explicit TrtEngine(const std::string& path, bool copy_outputs) : copy_outputs_(copy_outputs) {
        initLibNvInferPlugins(&gLogger, "");
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("failed to open engine: " + path);
        f.seekg(0, std::ios::end); size_t size = static_cast<size_t>(f.tellg()); f.seekg(0);
        std::vector<char> data(size); f.read(data.data(), size);
        runtime_.reset(nvinfer1::createInferRuntime(gLogger));
        if (!runtime_) throw std::runtime_error("createInferRuntime failed");
        engine_.reset(runtime_->deserializeCudaEngine(data.data(), data.size()));
        if (!engine_) throw std::runtime_error("deserializeCudaEngine failed");
        ctx_.reset(engine_->createExecutionContext());
        if (!ctx_) throw std::runtime_error("createExecutionContext failed");
        cudaStreamCreate(&stream_);
        int nb = engine_->getNbIOTensors();
        for (int i = 0; i < nb; ++i) {
            TrtBinding b;
            b.name = engine_->getIOTensorName(i);
            b.mode = engine_->getTensorIOMode(b.name.c_str());
            b.dtype = engine_->getTensorDataType(b.name.c_str());
            b.dims = engine_->getTensorShape(b.name.c_str());
            b.bytes = vol(b.dims) * trt_dtype_size(b.dtype);
            cudaMalloc(&b.device, b.bytes);
            if (b.mode == nvinfer1::TensorIOMode::kOUTPUT && copy_outputs_) b.host.resize(b.bytes);
            ctx_->setTensorAddress(b.name.c_str(), b.device);
            if (b.mode == nvinfer1::TensorIOMode::kINPUT) input_index_ = bindings_.size();
            bindings_.push_back(std::move(b));
        }
        if (input_index_ >= bindings_.size()) throw std::runtime_error("engine has no input");
    }
    ~TrtEngine() {
        for (auto& b : bindings_) if (b.device) cudaFree(b.device);
        if (stream_) cudaStreamDestroy(stream_);
    }
    const TrtBinding& input() const { return bindings_.at(input_index_); }
    double run(const uint8_t* input_data, size_t input_bytes) {
        const auto& in = input();
        if (input_bytes != in.bytes) {
            std::ostringstream os; os << "TRT input byte mismatch: got " << input_bytes << " expected " << in.bytes << " for " << in.name << " dtype=" << trt_dtype_name(in.dtype);
            throw std::runtime_error(os.str());
        }
        auto t0 = Clock::now();
        cudaMemcpyAsync(in.device, input_data, input_bytes, cudaMemcpyHostToDevice, stream_);
#if NV_TENSORRT_MAJOR >= 10
        bool ok = ctx_->enqueueV3(stream_);
#else
        std::vector<void*> ptrs;
        for (auto& b : bindings_) ptrs.push_back(b.device);
        bool ok = ctx_->enqueueV2(ptrs.data(), stream_, nullptr);
#endif
        if (!ok) throw std::runtime_error("TensorRT enqueue failed");
        if (copy_outputs_) {
            for (auto& b : bindings_) {
                if (b.mode == nvinfer1::TensorIOMode::kOUTPUT) cudaMemcpyAsync(b.host.data(), b.device, b.bytes, cudaMemcpyDeviceToHost, stream_);
            }
        }
        cudaStreamSynchronize(stream_);
        auto t1 = Clock::now();
        return ms_since(t0, t1);
    }
    std::string input_desc() const {
        const auto& i = input();
        return i.name + " " + dims_to_string(i.dims) + " " + trt_dtype_name(i.dtype) + " " + std::to_string(i.bytes) + "B";
    }
private:
    struct RuntimeDel { void operator()(nvinfer1::IRuntime* p) const { delete p; } };
    struct EngineDel { void operator()(nvinfer1::ICudaEngine* p) const { delete p; } };
    struct ContextDel { void operator()(nvinfer1::IExecutionContext* p) const { delete p; } };
    std::unique_ptr<nvinfer1::IRuntime, RuntimeDel> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, EngineDel> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, ContextDel> ctx_;
    cudaStream_t stream_{};
    std::vector<TrtBinding> bindings_;
    size_t input_index_ = static_cast<size_t>(-1);
    bool copy_outputs_ = true;
};

class HailoPart1 {
public:
    explicit HailoPart1(const std::string& hef_path) {
        auto vdev = hailort::VDevice::create();
        if (!vdev) throw std::runtime_error("VDevice::create failed status=" + std::to_string(vdev.status()));
        vdev_ = std::move(vdev.value());
        auto hef = hailort::Hef::create(hef_path);
        if (!hef) throw std::runtime_error("Hef::create failed status=" + std::to_string(hef.status()));
        auto cfg = hef->create_configure_params(HAILO_STREAM_INTERFACE_PCIE);
        if (!cfg) throw std::runtime_error("create_configure_params failed");
        auto ngs = vdev_->configure(hef.value(), cfg.value());
        if (!ngs) throw std::runtime_error("configure failed status=" + std::to_string(ngs.status()));
        if (ngs->empty()) throw std::runtime_error("no network groups");
        ng_ = std::move(ngs->front());
        auto streams = hailort::VStreamsBuilder::create_vstreams(*ng_, true, HAILO_FORMAT_TYPE_AUTO);
        if (!streams) throw std::runtime_error("create_vstreams failed status=" + std::to_string(streams.status()));
        inputs_ = std::move(streams->first);
        outputs_ = std::move(streams->second);
        if (inputs_.empty() || outputs_.empty()) throw std::runtime_error("missing Hailo input/output vstream");
        in_bytes_ = inputs_.front().get_frame_size();
        out_bytes_ = outputs_.front().get_frame_size();
        in_info_name_ = inputs_.front().get_info().name;
        out_info_name_ = outputs_.front().get_info().name;
    }
    size_t input_bytes() const { return in_bytes_; }
    size_t output_bytes() const { return out_bytes_; }
    std::string desc() const { return in_info_name_ + " -> " + out_info_name_ + " out_bytes=" + std::to_string(out_bytes_); }
    void run(const std::vector<uint8_t>& input, std::vector<uint8_t>& output) {
        if (input.size() != in_bytes_) throw std::runtime_error("Hailo input bytes mismatch");
        output.resize(out_bytes_);
        auto st = inputs_.front().write(hailort::MemoryView(const_cast<uint8_t*>(input.data()), input.size()));
        if (st != HAILO_SUCCESS) throw std::runtime_error("Hailo input write failed status=" + std::to_string(st));
        st = outputs_.front().read(hailort::MemoryView(output.data(), output.size()));
        if (st != HAILO_SUCCESS) throw std::runtime_error("Hailo output read failed status=" + std::to_string(st));
    }
private:
    std::unique_ptr<hailort::VDevice> vdev_;
    std::shared_ptr<hailort::ConfiguredNetworkGroup> ng_;
    std::vector<hailort::InputVStream> inputs_;
    std::vector<hailort::OutputVStream> outputs_;
    size_t in_bytes_ = 0, out_bytes_ = 0;
    std::string in_info_name_, out_info_name_;
};

static std::vector<std::string> list_images(const Args& a) {
    std::vector<std::string> out;
    if (!a.image.empty()) out.push_back(a.image);
    if (!a.image_dir.empty()) {
        for (auto& p : fs::directory_iterator(a.image_dir)) {
            if (!p.is_regular_file()) continue;
            auto ext = p.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") out.push_back(p.path().string());
        }
        std::sort(out.begin(), out.end());
    }
    return out;
}

static std::vector<uint8_t> make_input(size_t bytes, const std::vector<std::string>& imgs, size_t idx) {
    std::vector<uint8_t> buf(bytes, 0);
#ifdef HAVE_OPENCV
    if (!imgs.empty()) {
        cv::Mat img = cv::imread(imgs[idx % imgs.size()]);
        if (!img.empty()) {
            // Fallback generic preprocessing: resize to 640x640 RGB/NHWC when byte size matches.
            cv::Mat resized, rgb;
            cv::resize(img, resized, cv::Size(640, 640));
            cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
            if (rgb.total() * rgb.elemSize() == bytes) {
                std::memcpy(buf.data(), rgb.data, bytes);
            }
        }
    }
#else
    (void)imgs; (void)idx;
#endif
    return buf;
}

static double mean(const std::vector<double>& v) { if (v.empty()) return 0.0; return std::accumulate(v.begin(), v.end(), 0.0) / double(v.size()); }

static void write_json(const std::string& path, const Args& args, const std::string& hdesc, const std::string& tdesc,
                       const std::vector<double>& p1, const std::vector<double>& p2, double makespan_ms, int frames, int warmup) {
    double p1m = mean(p1), p2m = mean(p2);
    double fps = frames > 0 && makespan_ms > 0 ? double(frames) * 1000.0 / makespan_ms : 0.0;
    double paper_cycle = std::max(p1m, p2m);
    std::ofstream f(path);
    f << "{\n";
    f << "  \"schema\": \"onnx-splitpoint/native-hailo-trt-fifo-report\",\n";
    f << "  \"schema_version\": 1,\n";
    f << "  \"hef\": \"" << args.hef << "\",\n";
    f << "  \"engine\": \"" << args.engine << "\",\n";
    f << "  \"hailo\": \"" << hdesc << "\",\n";
    f << "  \"tensorrt_input\": \"" << tdesc << "\",\n";
    f << "  \"warmup\": " << warmup << ",\n";
    f << "  \"frames\": " << frames << ",\n";
    f << "  \"queue_depth\": " << args.queue_depth << ",\n";
    f << "  \"copy_outputs\": " << (args.copy_outputs ? "true" : "false") << ",\n";
    f << "  \"p1_ms_mean\": " << p1m << ",\n";
    f << "  \"p2_ms_mean\": " << p2m << ",\n";
    f << "  \"makespan_ms\": " << makespan_ms << ",\n";
    f << "  \"fps_makespan\": " << fps << ",\n";
    f << "  \"paper_equivalent_cycle_ms\": " << paper_cycle << ",\n";
    f << "  \"paper_equivalent_fps\": " << (paper_cycle > 0 ? 1000.0 / paper_cycle : 0.0) << "\n";
    f << "}\n";
}

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        HailoPart1 hailo(args.hef);
        TrtEngine trt(args.engine, args.copy_outputs);
        std::cerr << "[native-fifo] Hailo: " << hailo.desc() << "\n";
        std::cerr << "[native-fifo] TRT input: " << trt.input_desc() << "\n";
        if (hailo.output_bytes() != trt.input().bytes) {
            std::ostringstream os; os << "Hailo output bytes " << hailo.output_bytes() << " != TRT input bytes " << trt.input().bytes << ". Use a matching bridge engine.";
            throw std::runtime_error(os.str());
        }
        std::vector<std::string> images = list_images(args);
        std::vector<std::vector<uint8_t>> inputs;
        int total = args.warmup + args.frames;
        inputs.reserve(total);
        for (int i = 0; i < total; ++i) inputs.push_back(make_input(hailo.input_bytes(), images, size_t(i)));
        BoundedQueue queue(size_t(args.queue_depth));
        std::atomic<bool> producer_done{false};
        std::exception_ptr eptr = nullptr;
        std::vector<double> p1_ms, p2_ms;
        p1_ms.reserve(args.frames); p2_ms.reserve(args.frames);
        auto measured_start = Clock::now();
        auto measured_end = measured_start;

        std::thread prod([&](){
            try {
                for (int i = 0; i < total; ++i) {
                    auto out = std::make_shared<std::vector<uint8_t>>();
                    auto t0 = Clock::now();
                    hailo.run(inputs[size_t(i)], *out);
                    auto t1 = Clock::now();
                    if (i >= args.warmup) p1_ms.push_back(ms_since(t0, t1));
                    if (i == args.warmup) measured_start = Clock::now();
                    queue.push(out);
                }
                producer_done = true; queue.close();
            } catch (...) { eptr = std::current_exception(); producer_done = true; queue.close(); }
        });
        std::thread cons([&](){
            try {
                int idx = 0;
                while (true) {
                    auto item = queue.pop();
                    if (!item) break;
                    auto t0 = Clock::now();
                    trt.run(item->data(), item->size());
                    auto t1 = Clock::now();
                    if (idx >= args.warmup) { p2_ms.push_back(ms_since(t0, t1)); measured_end = t1; }
                    ++idx;
                }
            } catch (...) { eptr = std::current_exception(); queue.close(); }
        });
        prod.join(); cons.join();
        if (eptr) std::rethrow_exception(eptr);
        double makespan = ms_since(measured_start, measured_end);
        write_json(args.out_json, args, hailo.desc(), trt.input_desc(), p1_ms, p2_ms, makespan, int(p2_ms.size()), args.warmup);
        std::cout << "[native-fifo] p1_ms=" << mean(p1_ms) << " p2_ms=" << mean(p2_ms) << " fps=" << (p2_ms.empty() ? 0.0 : double(p2_ms.size())*1000.0/makespan) << " paper_fps=" << (1000.0/std::max(mean(p1_ms), mean(p2_ms))) << " out=" << args.out_json << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[native-fifo][error] " << e.what() << "\n";
        return 2;
    }
}
