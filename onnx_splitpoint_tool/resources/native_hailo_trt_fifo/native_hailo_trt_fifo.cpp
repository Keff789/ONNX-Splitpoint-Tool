// Generic HailoRT -> TensorRT FIFO fastpath for ONNX-Splitpoint-Tool.
//
// This runner is intentionally small and manifest-driven. It is meant as the
// native counterpart to the generated Python split runner for Hailo->TensorRT
// performance investigations. It does not implement task postprocessing; it
// measures the hot loop: preprocess/input staging, Hailo P1, FIFO handoff,
// TensorRT P2, and pipeline throughput.

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
#include <hailo/hailort.hpp>

#if SPLITPOINT_WITH_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using Ms = std::chrono::duration<double, std::milli>;

class TrtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char *msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TRT] " << msg << "\n";
        }
    }
};

static TrtLogger gLogger;

static std::vector<char> read_file(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("failed to open file: " + path);
    f.seekg(0, std::ios::end);
    const auto n = static_cast<size_t>(f.tellg());
    f.seekg(0, std::ios::beg);
    std::vector<char> data(n);
    f.read(data.data(), static_cast<std::streamsize>(n));
    return data;
}

static size_t dtype_size(nvinfer1::DataType t) {
    switch (t) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT8: return 1;
        case nvinfer1::DataType::kUINT8: return 1;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kINT64: return 8;
        case nvinfer1::DataType::kBOOL: return 1;
#if NV_TENSORRT_MAJOR >= 10
        case nvinfer1::DataType::kBF16: return 2;
#endif
        default: return 4;
    }
}

static std::string dtype_name(nvinfer1::DataType t) {
    switch (t) {
        case nvinfer1::DataType::kFLOAT: return "float32";
        case nvinfer1::DataType::kHALF: return "float16";
        case nvinfer1::DataType::kINT8: return "int8";
        case nvinfer1::DataType::kUINT8: return "uint8";
        case nvinfer1::DataType::kINT32: return "int32";
        case nvinfer1::DataType::kINT64: return "int64";
        case nvinfer1::DataType::kBOOL: return "bool";
#if NV_TENSORRT_MAJOR >= 10
        case nvinfer1::DataType::kBF16: return "bf16";
#endif
        default: return "unknown";
    }
}

static int64_t volume(const nvinfer1::Dims &d) {
    int64_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) v *= std::max(1, d.d[i]);
    return v;
}

struct Args {
    std::string hef;
    std::string engine;
    std::string image;
    std::string out_json = "native_hailo_trt_fifo_results.json";
    std::string device_id;
    int warmup = 8;
    int frames = 64;
    int queue_depth = 2;
    int repeat_input = 1;
    bool copy_outputs = true;
    bool no_preprocess = false;
    std::string boundary_mode = "raw_uint8_hailo";
};

static Args parse_args(int argc, char **argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&](const char *name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (k == "--hef") a.hef = need("--hef");
        else if (k == "--engine") a.engine = need("--engine");
        else if (k == "--image") a.image = need("--image");
        else if (k == "--out-json") a.out_json = need("--out-json");
        else if (k == "--device-id") a.device_id = need("--device-id");
        else if (k == "--warmup") a.warmup = std::stoi(need("--warmup"));
        else if (k == "--frames") a.frames = std::stoi(need("--frames"));
        else if (k == "--queue-depth") a.queue_depth = std::stoi(need("--queue-depth"));
        else if (k == "--repeat-input") a.repeat_input = std::stoi(need("--repeat-input"));
        else if (k == "--boundary-mode") a.boundary_mode = need("--boundary-mode");
        else if (k == "--no-copy-outputs") a.copy_outputs = false;
        else if (k == "--copy-outputs") a.copy_outputs = true;
        else if (k == "--no-preprocess") a.no_preprocess = true;
        else if (k == "--help" || k == "-h") {
            std::cout << "Usage: " << argv[0] << " --hef part1.hef --engine part2.engine --image img.jpg [--frames 64 --warmup 8 --queue-depth 2]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + k);
        }
    }
    if (a.hef.empty()) throw std::runtime_error("--hef is required");
    if (a.engine.empty()) throw std::runtime_error("--engine is required");
    if (a.image.empty()) throw std::runtime_error("--image is required");
    a.queue_depth = std::max(1, a.queue_depth);
    a.frames = std::max(1, a.frames);
    a.warmup = std::max(0, a.warmup);
    return a;
}

template <typename T>
class BlockingQueue {
public:
    explicit BlockingQueue(size_t capacity) : capacity_(std::max<size_t>(1, capacity)) {}

    void push(T item) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_not_full_.wait(lk, [&]{ return q_.size() < capacity_ || closed_; });
        if (closed_) return;
        q_.push(std::move(item));
        cv_not_empty_.notify_one();
    }

    bool pop(T &out) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_not_empty_.wait(lk, [&]{ return !q_.empty() || closed_; });
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop();
        cv_not_full_.notify_one();
        return true;
    }

    void close() {
        std::lock_guard<std::mutex> lk(mu_);
        closed_ = true;
        cv_not_empty_.notify_all();
        cv_not_full_.notify_all();
    }

private:
    size_t capacity_;
    std::queue<T> q_;
    bool closed_ = false;
    std::mutex mu_;
    std::condition_variable cv_not_empty_, cv_not_full_;
};

struct TensorBuffer {
    std::string name;
    nvinfer1::Dims dims{};
    nvinfer1::DataType dtype{};
    size_t bytes = 0;
    void *host = nullptr;
    void *device = nullptr;
};

class NativeTrtEngine {
public:
    explicit NativeTrtEngine(const std::string &engine_path) {
        initLibNvInferPlugins(&gLogger, "");
        auto blob = read_file(engine_path);
        runtime_.reset(nvinfer1::createInferRuntime(gLogger));
        if (!runtime_) throw std::runtime_error("createInferRuntime failed");
        engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
        if (!engine_) throw std::runtime_error("deserializeCudaEngine failed");
        context_.reset(engine_->createExecutionContext());
        if (!context_) throw std::runtime_error("createExecutionContext failed");
        if (cudaStreamCreate(&stream_) != cudaSuccess) throw std::runtime_error("cudaStreamCreate failed");
        allocate_buffers();
    }

    ~NativeTrtEngine() {
        for (auto &b : bindings_) {
            if (b.host) cudaFreeHost(b.host);
            if (b.device) cudaFree(b.device);
        }
        if (stream_) cudaStreamDestroy(stream_);
    }

    TensorBuffer &input() {
        if (input_index_ < 0) throw std::runtime_error("no TensorRT input binding");
        return bindings_.at(static_cast<size_t>(input_index_));
    }

    double run_from_host(const void *src, size_t src_bytes, bool copy_outputs) {
        TensorBuffer &inp = input();
        if (src_bytes > inp.bytes) throw std::runtime_error("source input is larger than TensorRT input buffer");
        auto t0 = Clock::now();
        auto e = cudaMemcpyAsync(inp.device, src, src_bytes, cudaMemcpyHostToDevice, stream_);
        if (e != cudaSuccess) throw std::runtime_error("cudaMemcpyAsync H2D failed");
        if (!context_->enqueueV3(stream_)) throw std::runtime_error("TensorRT enqueueV3 failed");
        if (copy_outputs) {
            for (auto &b : bindings_) {
                if (b.name == inp.name) continue;
                e = cudaMemcpyAsync(b.host, b.device, b.bytes, cudaMemcpyDeviceToHost, stream_);
                if (e != cudaSuccess) throw std::runtime_error("cudaMemcpyAsync D2H failed");
            }
        }
        if (cudaStreamSynchronize(stream_) != cudaSuccess) throw std::runtime_error("cudaStreamSynchronize failed");
        auto t1 = Clock::now();
        return Ms(t1 - t0).count();
    }

    std::string input_dtype() const { return dtype_name(bindings_.at(static_cast<size_t>(input_index_)).dtype); }
    size_t input_bytes() const { return bindings_.at(static_cast<size_t>(input_index_)).bytes; }

private:
    struct InferDeleter {
        template <typename T>
        void operator()(T *obj) const { delete obj; }
    };
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    cudaStream_t stream_{};
    std::vector<TensorBuffer> bindings_;
    int input_index_ = -1;

    void allocate_buffers() {
        int nb = engine_->getNbIOTensors();
        bindings_.reserve(static_cast<size_t>(nb));
        for (int i = 0; i < nb; ++i) {
            const char *cname = engine_->getIOTensorName(i);
            if (!cname) continue;
            TensorBuffer b;
            b.name = cname;
            b.dims = engine_->getTensorShape(cname);
            b.dtype = engine_->getTensorDataType(cname);
            b.bytes = static_cast<size_t>(std::max<int64_t>(1, volume(b.dims))) * dtype_size(b.dtype);
            if (cudaMallocHost(&b.host, b.bytes) != cudaSuccess) throw std::runtime_error("cudaMallocHost failed");
            if (cudaMalloc(&b.device, b.bytes) != cudaSuccess) throw std::runtime_error("cudaMalloc failed");
            if (!context_->setTensorAddress(cname, b.device)) throw std::runtime_error("setTensorAddress failed for " + b.name);
            if (engine_->getTensorIOMode(cname) == nvinfer1::TensorIOMode::kINPUT) input_index_ = static_cast<int>(bindings_.size());
            bindings_.push_back(b);
        }
        if (input_index_ < 0) throw std::runtime_error("no TensorRT input found");
    }
};

#if SPLITPOINT_WITH_OPENCV
static std::vector<uint8_t> load_image_rgb_letterbox(const std::string &path, int dst_w, int dst_h) {
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty()) throw std::runtime_error("failed to read image: " + path);
    double scale = std::min(dst_w / static_cast<double>(img.cols), dst_h / static_cast<double>(img.rows));
    int new_w = std::max(1, static_cast<int>(std::round(img.cols * scale)));
    int new_h = std::max(1, static_cast<int>(std::round(img.rows * scale)));
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h));
    cv::Mat canvas(dst_h, dst_w, CV_8UC3, cv::Scalar(0,0,0));
    int left = (dst_w - new_w) / 2;
    int top = (dst_h - new_h) / 2;
    resized.copyTo(canvas(cv::Rect(left, top, new_w, new_h)));
    cv::Mat rgb;
    cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
    std::vector<uint8_t> out(rgb.total() * rgb.elemSize());
    std::memcpy(out.data(), rgb.data, out.size());
    return out;
}
#endif

class HailoP1 {
public:
    HailoP1(const std::string &hef_path, const std::string &device_id) {
        auto vdev_exp = device_id.empty() ? hailort::VDevice::create() : hailort::VDevice::create({device_id});
        if (!vdev_exp) throw std::runtime_error("Hailo VDevice::create failed");
        vdevice_ = std::move(vdev_exp.value());
        auto hef_exp = hailort::Hef::create(hef_path);
        if (!hef_exp) throw std::runtime_error("Hef::create failed");
        auto params_exp = hef_exp->create_configure_params(HAILO_STREAM_INTERFACE_PCIE);
        if (!params_exp) throw std::runtime_error("create_configure_params failed");
        auto ngs_exp = vdevice_->configure(hef_exp.value(), params_exp.value());
        if (!ngs_exp) throw std::runtime_error("VDevice::configure failed");
        if (ngs_exp->empty()) throw std::runtime_error("no Hailo network groups");
        network_group_ = std::move(ngs_exp->front());
        auto streams_exp = hailort::VStreamsBuilder::create_vstreams(*network_group_, true, HAILO_FORMAT_TYPE_AUTO);
        if (!streams_exp) throw std::runtime_error("VStreamsBuilder::create_vstreams failed");
        vstreams_ = std::move(streams_exp.value());
        if (vstreams_.first.empty() || vstreams_.second.empty()) throw std::runtime_error("missing Hailo vstreams");
        input_h_ = vstreams_.first.front().get_info().shape.height;
        input_w_ = vstreams_.first.front().get_info().shape.width;
        input_bytes_ = vstreams_.first.front().get_frame_size();
        output_bytes_ = vstreams_.second.front().get_frame_size();
        std::cerr << "[native-fastpath][hailo] input=" << input_w_ << "x" << input_h_ << " bytes=" << input_bytes_ << " output_bytes=" << output_bytes_ << "\n";
    }

    int input_w() const { return input_w_; }
    int input_h() const { return input_h_; }
    size_t input_bytes() const { return input_bytes_; }
    size_t output_bytes() const { return output_bytes_; }

    double infer(const std::vector<uint8_t> &input, std::vector<uint8_t> &output) {
        if (input.size() < input_bytes_) throw std::runtime_error("Hailo input too small");
        output.resize(output_bytes_);
        auto t0 = Clock::now();
        auto status = vstreams_.first.front().write(hailort::MemoryView(const_cast<uint8_t*>(input.data()), input_bytes_));
        if (HAILO_SUCCESS != status) throw std::runtime_error("Hailo input write failed");
        status = vstreams_.second.front().read(hailort::MemoryView(output.data(), output.size()));
        if (HAILO_SUCCESS != status) throw std::runtime_error("Hailo output read failed");
        auto t1 = Clock::now();
        return Ms(t1 - t0).count();
    }

private:
    std::unique_ptr<hailort::VDevice> vdevice_;
    std::shared_ptr<hailort::ConfiguredNetworkGroup> network_group_;
    std::pair<std::vector<hailort::InputVStream>, std::vector<hailort::OutputVStream>> vstreams_;
    int input_w_ = 0, input_h_ = 0;
    size_t input_bytes_ = 0, output_bytes_ = 0;
};

struct FramePacket {
    int index = 0;
    std::vector<uint8_t> boundary;
    double p1_ms = 0.0;
    Clock::time_point ready_time;
};

static double mean(const std::vector<double> &v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

static double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t idx = static_cast<size_t>(std::round((p / 100.0) * (v.size() - 1)));
    return v[std::min(idx, v.size()-1)];
}

static std::string json_escape(const std::string &s) {
    std::ostringstream o;
    for (char c : s) {
        switch (c) { case '"': o << "\\\""; break; case '\\': o << "\\\\"; break; case '\n': o << "\\n"; break; default: o << c; }
    }
    return o.str();
}

int main(int argc, char **argv) {
    try {
        Args args = parse_args(argc, argv);
        NativeTrtEngine trt(args.engine);
        HailoP1 hailo(args.hef, args.device_id);
        if (trt.input_dtype() != "uint8" && args.boundary_mode == "raw_uint8_hailo") {
            throw std::runtime_error("raw_uint8_hailo requires a TensorRT uint8 input engine; got " + trt.input_dtype());
        }
        if (hailo.output_bytes() > trt.input_bytes()) {
            throw std::runtime_error("Hailo output is larger than TensorRT input: " + std::to_string(hailo.output_bytes()) + " > " + std::to_string(trt.input_bytes()));
        }
#if SPLITPOINT_WITH_OPENCV
        auto input_img = args.no_preprocess ? std::vector<uint8_t>() : load_image_rgb_letterbox(args.image, hailo.input_w(), hailo.input_h());
#else
        std::vector<uint8_t> input_img;
#endif
        if (input_img.empty()) {
            input_img.assign(hailo.input_bytes(), 0);
            std::ifstream f(args.image, std::ios::binary);
            if (f) f.read(reinterpret_cast<char*>(input_img.data()), static_cast<std::streamsize>(input_img.size()));
        }

        const int total_frames = args.warmup + args.frames;
        BlockingQueue<FramePacket> q(static_cast<size_t>(args.queue_depth));
        std::atomic<bool> p1_done{false};
        std::vector<double> p1_ms, p2_ms, handoff_ms, latency_ms;
        p1_ms.reserve(args.frames); p2_ms.reserve(args.frames); handoff_ms.reserve(args.frames); latency_ms.reserve(args.frames);
        auto global_t0 = Clock::now();
        auto measured_t0 = global_t0;
        auto measured_t1 = global_t0;

        std::thread t1([&](){
            for (int i = 0; i < total_frames; ++i) {
                FramePacket pkt;
                pkt.index = i;
                pkt.p1_ms = hailo.infer(input_img, pkt.boundary);
                pkt.ready_time = Clock::now();
                q.push(std::move(pkt));
            }
            p1_done = true;
            q.close();
        });

        std::thread t2([&](){
            FramePacket pkt;
            int consumed = 0;
            while (q.pop(pkt)) {
                if (pkt.index == args.warmup) measured_t0 = Clock::now();
                auto start_p2 = Clock::now();
                double run_ms = trt.run_from_host(pkt.boundary.data(), pkt.boundary.size(), args.copy_outputs);
                auto end_p2 = Clock::now();
                if (pkt.index >= args.warmup) {
                    p1_ms.push_back(pkt.p1_ms);
                    p2_ms.push_back(run_ms);
                    handoff_ms.push_back(Ms(start_p2 - pkt.ready_time).count());
                    latency_ms.push_back(Ms(end_p2 - pkt.ready_time).count());
                    measured_t1 = end_p2;
                }
                consumed++;
            }
        });

        t1.join();
        t2.join();
        double elapsed_ms = Ms(measured_t1 - measured_t0).count();
        double fps = args.frames > 0 && elapsed_ms > 0.0 ? (1000.0 * args.frames / elapsed_ms) : 0.0;
        double p1_mean = mean(p1_ms);
        double p2_mean = mean(p2_ms);
        double handoff_mean = mean(handoff_ms);
        double p2_total = p2_mean + handoff_mean;
        double paper_cycle = std::max(p1_mean, p2_total);
        double paper_fps = paper_cycle > 0.0 ? 1000.0 / paper_cycle : 0.0;

        std::ofstream out(args.out_json);
        out << std::fixed << std::setprecision(6);
        out << "{\n";
        out << "  \"schema\": \"onnx-splitpoint/native-hailo-trt-fifo-results\",\n";
        out << "  \"schema_version\": 1,\n";
        out << "  \"hef\": \"" << json_escape(args.hef) << "\",\n";
        out << "  \"engine\": \"" << json_escape(args.engine) << "\",\n";
        out << "  \"frames\": " << args.frames << ",\n";
        out << "  \"warmup\": " << args.warmup << ",\n";
        out << "  \"queue_depth\": " << args.queue_depth << ",\n";
        out << "  \"boundary_mode\": \"" << json_escape(args.boundary_mode) << "\",\n";
        out << "  \"trt_input_dtype\": \"" << trt.input_dtype() << "\",\n";
        out << "  \"hailo_output_bytes\": " << hailo.output_bytes() << ",\n";
        out << "  \"trt_input_bytes\": " << trt.input_bytes() << ",\n";
        out << "  \"streaming_fps\": " << fps << ",\n";
        out << "  \"paper_equivalent_cycle_ms\": " << paper_cycle << ",\n";
        out << "  \"paper_equivalent_fps\": " << paper_fps << ",\n";
        out << "  \"p1_mean_ms\": " << p1_mean << ",\n";
        out << "  \"p2_run_mean_ms\": " << p2_mean << ",\n";
        out << "  \"handoff_queue_mean_ms\": " << handoff_mean << ",\n";
        out << "  \"p2_total_mean_ms\": " << p2_total << ",\n";
        out << "  \"latency_p50_ms\": " << percentile(latency_ms, 50) << ",\n";
        out << "  \"latency_p95_ms\": " << percentile(latency_ms, 95) << "\n";
        out << "}\n";
        out.close();

        std::cout << "[native-fastpath] frames=" << args.frames << " fps=" << fps
                  << " paper_fps=" << paper_fps << " p1=" << p1_mean
                  << " p2_run=" << p2_mean << " handoff=" << handoff_mean
                  << " trt_input_dtype=" << trt.input_dtype() << "\n";
        std::cout << "[native-fastpath] wrote " << args.out_json << "\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "[native-fastpath][error] " << e.what() << "\n";
        return 2;
    }
}
