// SPDX-License-Identifier: MIT
// Generic HailoRT -> TensorRT FIFO fastpath smoke runner.
// v59ap: first native C++ bridge used to separate Python mapping overhead from
// HailoRT/TensorRT runtime overhead. It is intentionally artifact-driven:
//   part1.hef + part2.engine + optional synthetic/input frame.
// It does not implement task postprocessing; it measures the native pipeline.

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <hailo/hailort.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

static double now_ms()
{
    using clock = std::chrono::steady_clock;
    static const auto t0 = clock::now();
    auto t = clock::now() - t0;
    return std::chrono::duration<double, std::milli>(t).count();
}

static std::string json_escape(const std::string &s)
{
    std::ostringstream o;
    for (char c : s) {
        switch (c) {
        case '"': o << "\\\""; break;
        case '\\': o << "\\\\"; break;
        case '\n': o << "\\n"; break;
        case '\r': o << "\\r"; break;
        case '\t': o << "\\t"; break;
        default: o << c; break;
        }
    }
    return o.str();
}

struct Stats {
    double mean = 0.0;
    double min = 0.0;
    double max = 0.0;
};

static Stats summarize(const std::vector<double> &xs)
{
    Stats s;
    if (xs.empty()) return s;
    s.min = *std::min_element(xs.begin(), xs.end());
    s.max = *std::max_element(xs.begin(), xs.end());
    s.mean = std::accumulate(xs.begin(), xs.end(), 0.0) / double(xs.size());
    return s;
}

class TrtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char *msg) noexcept override
    {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[trt] " << msg << std::endl;
        }
    }
};

static size_t dtype_size(nvinfer1::DataType t)
{
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

static std::string dtype_name(nvinfer1::DataType t)
{
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

static size_t dims_volume(const nvinfer1::Dims &d)
{
    size_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) {
        int64_t x = d.d[i];
        if (x < 0) throw std::runtime_error("dynamic TensorRT shape is not supported by this smoke runner");
        v *= static_cast<size_t>(x);
    }
    return v;
}

static void check_cuda(cudaError_t err, const char *what)
{
    if (err != cudaSuccess) {
        std::ostringstream ss;
        ss << what << " failed: " << cudaGetErrorString(err);
        throw std::runtime_error(ss.str());
    }
}

struct TrtTiming {
    double h2d_ms = 0.0;
    double enqueue_ms = 0.0;
    double d2h_sync_ms = 0.0;
    double total_ms = 0.0;
};

class TrtEngine {
public:
    explicit TrtEngine(const std::string &engine_path, bool copy_outputs)
        : copy_outputs_(copy_outputs)
    {
        std::ifstream f(engine_path, std::ios::binary);
        if (!f) throw std::runtime_error("failed to open TensorRT engine: " + engine_path);
        f.seekg(0, std::ios::end);
        const size_t n = static_cast<size_t>(f.tellg());
        f.seekg(0, std::ios::beg);
        std::vector<char> blob(n);
        f.read(blob.data(), static_cast<std::streamsize>(blob.size()));

        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        if (!runtime_) throw std::runtime_error("createInferRuntime failed");
        engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
        if (!engine_) throw std::runtime_error("deserializeCudaEngine failed");
        context_.reset(engine_->createExecutionContext());
        if (!context_) throw std::runtime_error("createExecutionContext failed");
        check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate");

        const int nb = engine_->getNbIOTensors();
        for (int i = 0; i < nb; ++i) {
            const char *name_c = engine_->getIOTensorName(i);
            std::string name(name_c ? name_c : "");
            nvinfer1::Dims shape = engine_->getTensorShape(name.c_str());
            nvinfer1::DataType dt = engine_->getTensorDataType(name.c_str());
            size_t bytes = dims_volume(shape) * dtype_size(dt);
            void *dev = nullptr;
            check_cuda(cudaMalloc(&dev, bytes), "cudaMalloc tensor");
            dev_ptrs_[name] = dev;
            bytes_[name] = bytes;
            dtypes_[name] = dt;
            if (engine_->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT) {
                if (!input_name_.empty()) throw std::runtime_error("only one TensorRT input is supported in this smoke runner");
                input_name_ = name;
                input_bytes_ = bytes;
                input_dtype_ = dt;
            } else {
                if (copy_outputs_) {
                    void *host = nullptr;
                    check_cuda(cudaMallocHost(&host, bytes), "cudaMallocHost output");
                    host_outs_[name] = host;
                }
            }
            if (!context_->setTensorAddress(name.c_str(), dev)) {
                throw std::runtime_error("setTensorAddress failed for " + name);
            }
        }
        if (input_name_.empty()) throw std::runtime_error("TensorRT engine has no input");
    }

    ~TrtEngine()
    {
        for (auto &kv : host_outs_) cudaFreeHost(kv.second);
        for (auto &kv : dev_ptrs_) cudaFree(kv.second);
        if (stream_) cudaStreamDestroy(stream_);
    }

    const std::string &input_name() const { return input_name_; }
    size_t input_bytes() const { return input_bytes_; }
    std::string input_dtype_name() const { return dtype_name(input_dtype_); }

    TrtTiming run(const void *input, size_t input_bytes)
    {
        if (input_bytes != input_bytes_) {
            std::ostringstream ss;
            ss << "TensorRT input bytes mismatch: got " << input_bytes << " expected " << input_bytes_;
            throw std::runtime_error(ss.str());
        }
        TrtTiming t;
        const double t0 = now_ms();
        check_cuda(cudaMemcpyAsync(dev_ptrs_.at(input_name_), input, input_bytes_, cudaMemcpyHostToDevice, stream_), "cudaMemcpyAsync H2D");
        const double t1 = now_ms();
        if (!context_->enqueueV3(stream_)) throw std::runtime_error("TensorRT enqueueV3 failed");
        const double t2 = now_ms();
        if (copy_outputs_) {
            for (auto &kv : host_outs_) {
                check_cuda(cudaMemcpyAsync(kv.second, dev_ptrs_.at(kv.first), bytes_.at(kv.first), cudaMemcpyDeviceToHost, stream_), "cudaMemcpyAsync D2H");
            }
        }
        check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
        const double t3 = now_ms();
        t.h2d_ms = t1 - t0;
        t.enqueue_ms = t2 - t1;
        t.d2h_sync_ms = t3 - t2;
        t.total_ms = t3 - t0;
        return t;
    }

private:
    struct TrtDeleter {
        template <typename T> void operator()(T *p) const { delete p; }
    };
    TrtLogger logger_;
    std::unique_ptr<nvinfer1::IRuntime, TrtDeleter> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter> context_;
    cudaStream_t stream_ = nullptr;
    bool copy_outputs_ = true;
    std::map<std::string, void *> dev_ptrs_;
    std::map<std::string, void *> host_outs_;
    std::map<std::string, size_t> bytes_;
    std::map<std::string, nvinfer1::DataType> dtypes_;
    std::string input_name_;
    size_t input_bytes_ = 0;
    nvinfer1::DataType input_dtype_ = nvinfer1::DataType::kFLOAT;
};

class HailoPart1 {
public:
    HailoPart1(const std::string &hef_path, const std::string &device_id)
    {
        auto vdev = device_id.empty() ? hailort::VDevice::create() : hailort::VDevice::create({device_id});
        if (!vdev) throw std::runtime_error("VDevice::create failed status=" + std::to_string(vdev.status()));
        vdevice_ = std::move(vdev.value());

        auto hef = hailort::Hef::create(hef_path);
        if (!hef) throw std::runtime_error("Hef::create failed status=" + std::to_string(hef.status()));
        auto cfg = hef->create_configure_params(HAILO_STREAM_INTERFACE_PCIE);
        if (!cfg) throw std::runtime_error("create_configure_params failed status=" + std::to_string(cfg.status()));
        auto ngs = vdevice_->configure(hef.value(), cfg.value());
        if (!ngs) throw std::runtime_error("VDevice::configure failed status=" + std::to_string(ngs.status()));
        if (ngs->empty()) throw std::runtime_error("no Hailo network groups");
        network_group_ = std::move(ngs->front());

        auto vstreams = hailort::VStreamsBuilder::create_vstreams(*network_group_, true, HAILO_FORMAT_TYPE_AUTO);
        if (!vstreams) throw std::runtime_error("create_vstreams failed status=" + std::to_string(vstreams.status()));
        vstreams_ = std::move(vstreams.value());
        if (vstreams_.first.size() != 1 || vstreams_.second.size() != 1) {
            std::ostringstream ss;
            ss << "this smoke runner currently supports one input and one output vstream, got inputs="
               << vstreams_.first.size() << " outputs=" << vstreams_.second.size();
            throw std::runtime_error(ss.str());
        }
        input_bytes_ = vstreams_.first.front().get_frame_size();
        output_bytes_ = vstreams_.second.front().get_frame_size();
    }

    size_t input_bytes() const { return input_bytes_; }
    size_t output_bytes() const { return output_bytes_; }

    double run(const uint8_t *input, uint8_t *output)
    {
        const double t0 = now_ms();
        auto st = vstreams_.first.front().write(hailort::MemoryView(const_cast<uint8_t *>(input), input_bytes_));
        if (HAILO_SUCCESS != st) throw std::runtime_error("Hailo input write failed status=" + std::to_string(st));
        st = vstreams_.second.front().read(hailort::MemoryView(output, output_bytes_));
        if (HAILO_SUCCESS != st) throw std::runtime_error("Hailo output read failed status=" + std::to_string(st));
        return now_ms() - t0;
    }

private:
    std::unique_ptr<hailort::VDevice> vdevice_;
    std::shared_ptr<hailort::ConfiguredNetworkGroup> network_group_;
    std::pair<std::vector<hailort::InputVStream>, std::vector<hailort::OutputVStream>> vstreams_;
    size_t input_bytes_ = 0;
    size_t output_bytes_ = 0;
};

struct Args {
    std::string hef;
    std::string engine;
    std::string out_json = "native_fifo_result.json";
    std::string device_id;
    int frames = 64;
    int warmup = 8;
    int queue_depth = 2;
    bool copy_outputs = true;
    bool print_json = true;
};

static Args parse_args(int argc, char **argv)
{
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + k);
            return argv[++i];
        };
        if (k == "--hef") a.hef = next();
        else if (k == "--engine") a.engine = next();
        else if (k == "--out-json") a.out_json = next();
        else if (k == "--device-id") a.device_id = next();
        else if (k == "--frames") a.frames = std::stoi(next());
        else if (k == "--warmup") a.warmup = std::stoi(next());
        else if (k == "--queue-depth") a.queue_depth = std::stoi(next());
        else if (k == "--no-copy-outputs") a.copy_outputs = false;
        else if (k == "--quiet") a.print_json = false;
        else if (k == "--help" || k == "-h") {
            std::cout << "Usage: " << argv[0] << " --hef part1.hef --engine part2.engine [--frames N --warmup N --queue-depth N --device-id ID --no-copy-outputs --out-json file]\n";
            std::exit(0);
        }
        else throw std::runtime_error("unknown argument: " + k);
    }
    if (a.hef.empty()) throw std::runtime_error("--hef is required");
    if (a.engine.empty()) throw std::runtime_error("--engine is required");
    if (a.frames < 1) a.frames = 1;
    if (a.warmup < 0) a.warmup = 0;
    if (a.queue_depth < 1) a.queue_depth = 1;
    return a;
}

struct Slot {
    std::vector<uint8_t> boundary;
    int frame_id = -1;
    double p1_ms = 0.0;
};

template <typename T>
class BlockingQueue {
public:
    void push(T v) {
        std::unique_lock<std::mutex> lk(m_);
        q_.push(std::move(v));
        cv_.notify_one();
    }
    T pop() {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&]{ return !q_.empty(); });
        T v = std::move(q_.front());
        q_.pop();
        return v;
    }
private:
    std::mutex m_;
    std::condition_variable cv_;
    std::queue<T> q_;
};

int main(int argc, char **argv)
{
    try {
        Args args = parse_args(argc, argv);
        HailoPart1 hailo(args.hef, args.device_id);
        TrtEngine trt(args.engine, args.copy_outputs);

        if (hailo.output_bytes() != trt.input_bytes()) {
            std::ostringstream ss;
            ss << "boundary bytes mismatch: Hailo output=" << hailo.output_bytes()
               << " TensorRT input=" << trt.input_bytes()
               << " TensorRT dtype=" << trt.input_dtype_name();
            throw std::runtime_error(ss.str());
        }

        const int total = args.warmup + args.frames;
        std::vector<uint8_t> input(hailo.input_bytes(), 0);
        std::vector<Slot> slots(static_cast<size_t>(args.queue_depth));
        for (auto &s : slots) s.boundary.resize(hailo.output_bytes());

        BlockingQueue<int> free_q, full_q;
        for (int i = 0; i < args.queue_depth; ++i) free_q.push(i);
        std::atomic<bool> producer_done{false};

        std::vector<double> p1_ms, p2_total_ms, trt_h2d_ms, trt_enqueue_ms, trt_d2h_sync_ms;
        p1_ms.reserve(args.frames);
        p2_total_ms.reserve(args.frames);
        double measured_start = 0.0, measured_end = 0.0;

        std::thread producer([&]{
            for (int f = 0; f < total; ++f) {
                int idx = free_q.pop();
                slots[idx].frame_id = f;
                slots[idx].p1_ms = hailo.run(input.data(), slots[idx].boundary.data());
                full_q.push(idx);
            }
            producer_done.store(true);
            full_q.push(-1);
        });

        std::thread consumer([&]{
            while (true) {
                int idx = full_q.pop();
                if (idx < 0) break;
                const int f = slots[idx].frame_id;
                const double t0 = now_ms();
                if (f == args.warmup) measured_start = t0;
                TrtTiming tt = trt.run(slots[idx].boundary.data(), slots[idx].boundary.size());
                const double t1 = now_ms();
                if (f >= args.warmup) {
                    p1_ms.push_back(slots[idx].p1_ms);
                    p2_total_ms.push_back(t1 - t0);
                    trt_h2d_ms.push_back(tt.h2d_ms);
                    trt_enqueue_ms.push_back(tt.enqueue_ms);
                    trt_d2h_sync_ms.push_back(tt.d2h_sync_ms);
                    measured_end = t1;
                }
                free_q.push(idx);
            }
        });

        producer.join();
        consumer.join();

        Stats sp1 = summarize(p1_ms);
        Stats sp2 = summarize(p2_total_ms);
        Stats sh2d = summarize(trt_h2d_ms);
        Stats senq = summarize(trt_enqueue_ms);
        Stats sd2h = summarize(trt_d2h_sync_ms);
        const double makespan_s = std::max(0.000001, (measured_end - measured_start) / 1000.0);
        const double fps_makespan = double(args.frames) / makespan_s;
        const double paper_cycle_ms = std::max(sp1.mean, sp2.mean);
        const double paper_fps = paper_cycle_ms > 0.0 ? 1000.0 / paper_cycle_ms : 0.0;

        std::ostringstream js;
        js << std::fixed << std::setprecision(6);
        js << "{\n";
        js << "  \"schema\": \"onnx-splitpoint/native-hailo-trt-fifo-fastpath\",\n";
        js << "  \"schema_version\": 1,\n";
        js << "  \"hef\": \"" << json_escape(args.hef) << "\",\n";
        js << "  \"engine\": \"" << json_escape(args.engine) << "\",\n";
        js << "  \"frames\": " << args.frames << ",\n";
        js << "  \"warmup\": " << args.warmup << ",\n";
        js << "  \"queue_depth\": " << args.queue_depth << ",\n";
        js << "  \"copy_outputs\": " << (args.copy_outputs ? "true" : "false") << ",\n";
        js << "  \"hailo_input_bytes\": " << hailo.input_bytes() << ",\n";
        js << "  \"boundary_bytes\": " << hailo.output_bytes() << ",\n";
        js << "  \"trt_input_name\": \"" << json_escape(trt.input_name()) << "\",\n";
        js << "  \"trt_input_dtype\": \"" << json_escape(trt.input_dtype_name()) << "\",\n";
        js << "  \"fps_makespan\": " << fps_makespan << ",\n";
        js << "  \"paper_equivalent_cycle_ms\": " << paper_cycle_ms << ",\n";
        js << "  \"paper_equivalent_fps\": " << paper_fps << ",\n";
        js << "  \"p1_ms\": {\"mean\": " << sp1.mean << ", \"min\": " << sp1.min << ", \"max\": " << sp1.max << "},\n";
        js << "  \"p2_thread_ms\": {\"mean\": " << sp2.mean << ", \"min\": " << sp2.min << ", \"max\": " << sp2.max << "},\n";
        js << "  \"trt_h2d_ms\": {\"mean\": " << sh2d.mean << ", \"min\": " << sh2d.min << ", \"max\": " << sh2d.max << "},\n";
        js << "  \"trt_enqueue_ms\": {\"mean\": " << senq.mean << ", \"min\": " << senq.min << ", \"max\": " << senq.max << "},\n";
        js << "  \"trt_d2h_sync_ms\": {\"mean\": " << sd2h.mean << ", \"min\": " << sd2h.min << ", \"max\": " << sd2h.max << "}\n";
        js << "}\n";

        std::ofstream out(args.out_json);
        out << js.str();
        out.close();
        if (args.print_json) std::cout << js.str();
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "[native-fifo][error] " << e.what() << std::endl;
        return 2;
    }
}
