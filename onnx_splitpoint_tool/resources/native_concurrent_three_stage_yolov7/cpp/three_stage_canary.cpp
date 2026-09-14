#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
#include <hailo/hailort.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
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
using Clock = std::chrono::steady_clock;

static inline double ms_between(const Clock::time_point &a, const Clock::time_point &b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static inline void check_cuda(cudaError_t e, const char *what) {
    if (e != cudaSuccess) {
        std::ostringstream oss;
        oss << what << " failed: " << cudaGetErrorString(e);
        throw std::runtime_error(oss.str());
    }
}

struct Logger final : public nvinfer1::ILogger {
    void log(Severity severity, const char *msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[three-stage][TRT] " << msg << std::endl;
        }
    }
};
static Logger g_logger;

static size_t dtype_size(nvinfer1::DataType t) {
    switch (t) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT8: return 1;
        case nvinfer1::DataType::kUINT8: return 1;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kBOOL: return 1;
#if NV_TENSORRT_MAJOR >= 10
        case nvinfer1::DataType::kINT64: return 8;
        case nvinfer1::DataType::kBF16: return 2;
#endif
        default: throw std::runtime_error("unsupported TensorRT dtype");
    }
}

static std::string dtype_name(nvinfer1::DataType t) {
    switch (t) {
        case nvinfer1::DataType::kFLOAT: return "float32";
        case nvinfer1::DataType::kHALF: return "float16";
        case nvinfer1::DataType::kINT8: return "int8";
        case nvinfer1::DataType::kUINT8: return "uint8";
        case nvinfer1::DataType::kINT32: return "int32";
        case nvinfer1::DataType::kBOOL: return "bool";
#if NV_TENSORRT_MAJOR >= 10
        case nvinfer1::DataType::kINT64: return "int64";
        case nvinfer1::DataType::kBF16: return "bf16";
#endif
        default: return "unknown";
    }
}

static size_t volume(const nvinfer1::Dims &d) {
    size_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) {
        if (d.d[i] < 0) {
            throw std::runtime_error("dynamic TensorRT dimensions are not supported");
        }
        v *= static_cast<size_t>(d.d[i]);
    }
    return v;
}

static std::string json_escape(const std::string &value) {
    std::ostringstream out;
    for (unsigned char c : value) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(c) << std::dec;
                } else {
                    out << static_cast<char>(c);
                }
        }
    }
    return out.str();
}

struct Summary {
    size_t count = 0;
    double mean = 0.0;
    double median = 0.0;
    double p05 = 0.0;
    double p95 = 0.0;
    double min = 0.0;
    double max = 0.0;
};

static double percentile_sorted(const std::vector<double> &sorted, double q) {
    if (sorted.empty()) return 0.0;
    if (sorted.size() == 1) return sorted.front();
    const double pos = static_cast<double>(sorted.size() - 1) * q;
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = static_cast<size_t>(std::ceil(pos));
    if (lo == hi) return sorted[lo];
    const double weight = pos - static_cast<double>(lo);
    return sorted[lo] * (1.0 - weight) + sorted[hi] * weight;
}

static Summary summarize(const std::vector<double> &values) {
    Summary s;
    s.count = values.size();
    if (values.empty()) return s;
    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    s.mean = std::accumulate(values.begin(), values.end(), 0.0) /
             static_cast<double>(values.size());
    s.median = percentile_sorted(sorted, 0.5);
    s.p05 = percentile_sorted(sorted, 0.05);
    s.p95 = percentile_sorted(sorted, 0.95);
    s.min = sorted.front();
    s.max = sorted.back();
    return s;
}

static void write_summary(std::ostream &out, const Summary &s) {
    out << "{\"count\":" << s.count
        << ",\"mean_ms\":" << s.mean
        << ",\"median_ms\":" << s.median
        << ",\"p05_ms\":" << s.p05
        << ",\"p95_ms\":" << s.p95
        << ",\"min_ms\":" << s.min
        << ",\"max_ms\":" << s.max << "}";
}

static void write_numeric_summary(std::ostream &out, const Summary &s) {
    out << "{\"count\":" << s.count
        << ",\"mean\":" << s.mean
        << ",\"median\":" << s.median
        << ",\"p05\":" << s.p05
        << ",\"p95\":" << s.p95
        << ",\"min\":" << s.min
        << ",\"max\":" << s.max << "}";
}

template <typename T>
class BlockingQueue {
public:
    explicit BlockingQueue(size_t capacity) : capacity_(capacity) {
        if (capacity_ == 0) throw std::runtime_error("queue capacity must be positive");
    }

    void push(T value) {
        std::unique_lock<std::mutex> lock(mu_);
        not_full_.wait(lock, [&] { return closed_ || q_.size() < capacity_; });
        if (closed_) throw std::runtime_error("push to closed queue");
        q_.push(std::move(value));
        not_empty_.notify_one();
    }

    bool pop(T &value) {
        std::unique_lock<std::mutex> lock(mu_);
        not_empty_.wait(lock, [&] { return closed_ || !q_.empty(); });
        if (q_.empty()) return false;
        value = std::move(q_.front());
        q_.pop();
        not_full_.notify_one();
        return true;
    }

    void close() {
        std::lock_guard<std::mutex> lock(mu_);
        closed_ = true;
        not_empty_.notify_all();
        not_full_.notify_all();
    }

private:
    size_t capacity_;
    std::queue<T> q_;
    std::mutex mu_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    bool closed_ = false;
};

extern "C" {

struct ThreeStageConfigV1 {
    uint32_t abi_version;
    uint32_t struct_size;
    const char *hef_path;
    const char *engine_path;
    const char *images_dir;
    const char *out_json;
    const char *device_id;
    int32_t frames;
    int32_t warmup;
    int32_t p1_queue_depth;
    int32_t post_queue_depth;
    int32_t letterbox_pad_value;
};

typedef int (*ThreeStagePostCallbackV1)(
    int32_t measured,
    int32_t sequence,
    int32_t image_index,
    int32_t original_width,
    int32_t original_height,
    const float *head_stride8,
    uint64_t head_stride8_elements,
    const float *head_stride16,
    uint64_t head_stride16_elements,
    const float *head_stride32,
    uint64_t head_stride32_elements,
    int32_t *detection_count,
    void *user_data);

size_t onnx_splitpoint_three_stage_config_size_v1();
int onnx_splitpoint_run_three_stage_v1(
    const ThreeStageConfigV1 *config,
    ThreeStagePostCallbackV1 callback,
    void *user_data,
    char *error_buffer,
    size_t error_buffer_size);
}

static void set_error(char *buffer, size_t size, const std::string &message) {
    if (!buffer || size == 0) return;
    const size_t count = std::min(size - 1, message.size());
    std::memcpy(buffer, message.data(), count);
    buffer[count] = '\0';
}

static std::vector<std::string> list_images(const std::string &path) {
    std::vector<std::string> images;
    fs::path root(path);
    if (fs::is_directory(root)) {
        for (const auto &entry : fs::directory_iterator(root)) {
            if (!entry.is_regular_file()) continue;
            std::string extension = entry.path().extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (extension == ".jpg" || extension == ".jpeg" || extension == ".png" || extension == ".bmp") {
                images.push_back(entry.path().string());
            }
        }
        std::sort(images.begin(), images.end());
    } else if (fs::is_regular_file(root)) {
        images.push_back(root.string());
    }
    if (images.empty()) throw std::runtime_error("no images found: " + path);
    return images;
}

static cv::Mat letterbox_rgb_uint8(const cv::Mat &bgr, int width, int height, int pad_value) {
    if (bgr.empty()) throw std::runtime_error("empty image");
    const double scale = std::min(width / static_cast<double>(bgr.cols),
                                  height / static_cast<double>(bgr.rows));
    const int resized_width = std::max(1, static_cast<int>(std::round(bgr.cols * scale)));
    const int resized_height = std::max(1, static_cast<int>(std::round(bgr.rows * scale)));
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(resized_width, resized_height));
    cv::Mat canvas(height, width, bgr.type(), cv::Scalar(pad_value, pad_value, pad_value));
    const int left = (width - resized_width) / 2;
    const int top = (height - resized_height) / 2;
    resized.copyTo(canvas(cv::Rect(left, top, resized_width, resized_height)));
    cv::Mat rgb;
    cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
    return rgb;
}

class HailoStage {
public:
    HailoStage(const std::string &hef_path, const std::string &device_id) {
        hailort::Expected<std::unique_ptr<hailort::VDevice>> vdevice_expected =
            device_id.empty()
                ? hailort::VDevice::create()
                : hailort::VDevice::create(std::vector<std::string>{device_id});
        if (!vdevice_expected) {
            std::ostringstream oss;
            oss << "VDevice::create failed status="
                << static_cast<int>(vdevice_expected.status())
                << " device_id='" << device_id << "'";
            throw std::runtime_error(oss.str());
        }
        vdevice_ = std::move(vdevice_expected.value());

        auto hef_expected = hailort::Hef::create(hef_path);
        if (!hef_expected) throw std::runtime_error("Hef::create failed: " + hef_path);
        auto configure_expected = hef_expected->create_configure_params(HAILO_STREAM_INTERFACE_PCIE);
        if (!configure_expected) throw std::runtime_error("create_configure_params failed");
        auto groups_expected = vdevice_->configure(hef_expected.value(), configure_expected.value());
        if (!groups_expected) throw std::runtime_error("VDevice configure failed");
        if (groups_expected->size() != 1) throw std::runtime_error("expected one Hailo network group");
        network_group_ = std::move(groups_expected->front());

        auto streams_expected = hailort::VStreamsBuilder::create_vstreams(
            *network_group_, true, HAILO_FORMAT_TYPE_UINT8);
        if (!streams_expected) throw std::runtime_error("create_vstreams failed");
        streams_ = std::move(streams_expected.value());
        if (streams_.first.size() != 1 || streams_.second.size() != 1) {
            std::ostringstream oss;
            oss << "three-stage canary requires exactly one Hailo input and one output VStream: inputs="
                << streams_.first.size() << " outputs=" << streams_.second.size();
            throw std::runtime_error(oss.str());
        }
        const auto input_info = streams_.first.front().get_info();
        input_height_ = input_info.shape.height;
        input_width_ = input_info.shape.width;
        input_features_ = input_info.shape.features;
        input_frame_size_ = streams_.first.front().get_frame_size();
        output_frame_size_ = streams_.second.front().get_frame_size();
        std::cerr << "[three-stage][hailo] input_shape=" << input_width_ << "x"
                  << input_height_ << "x" << input_features_
                  << " input_bytes=" << input_frame_size_
                  << " boundary_bytes=" << output_frame_size_ << std::endl;
    }

    int input_width() const { return input_width_; }
    int input_height() const { return input_height_; }
    size_t output_frame_size() const { return output_frame_size_; }

    void infer(const cv::Mat &rgb_uint8, std::vector<uint8_t> &output) {
        const size_t input_bytes = static_cast<size_t>(rgb_uint8.total() * rgb_uint8.elemSize());
        if (input_frame_size_ != input_bytes) {
            std::ostringstream oss;
            oss << "Hailo input frame size mismatch: runtime=" << input_frame_size_
                << " image=" << input_bytes;
            throw std::runtime_error(oss.str());
        }
        const uint8_t *source = rgb_uint8.ptr<uint8_t>(0);
        auto status = streams_.first.front().write(
            hailort::MemoryView(const_cast<uint8_t *>(source), input_bytes));
        if (status != HAILO_SUCCESS) throw std::runtime_error("Hailo input write failed");
        output.resize(output_frame_size_);
        status = streams_.second.front().read(hailort::MemoryView(output.data(), output.size()));
        if (status != HAILO_SUCCESS) throw std::runtime_error("Hailo output read failed");
    }

private:
    std::unique_ptr<hailort::VDevice> vdevice_;
    std::shared_ptr<hailort::ConfiguredNetworkGroup> network_group_;
    std::pair<std::vector<hailort::InputVStream>, std::vector<hailort::OutputVStream>> streams_;
    int input_width_ = 0;
    int input_height_ = 0;
    int input_features_ = 0;
    size_t input_frame_size_ = 0;
    size_t output_frame_size_ = 0;
};

struct TensorBinding {
    std::string name;
    bool is_input = false;
    nvinfer1::DataType dtype = nvinfer1::DataType::kFLOAT;
    nvinfer1::Dims dims{};
    size_t elements = 0;
    size_t bytes = 0;
    void *device = nullptr;
    void *input_host = nullptr;
};

struct PostSlot {
    float *head8 = nullptr;
    float *head16 = nullptr;
    float *head32 = nullptr;
};

class TRTStage {
public:
    TRTStage(const std::string &engine_path, int post_slot_count)
        : post_slots_(static_cast<size_t>(post_slot_count)) {
        initLibNvInferPlugins(&g_logger, "");
        std::ifstream stream(engine_path, std::ios::binary);
        if (!stream) throw std::runtime_error("failed to open engine: " + engine_path);
        stream.seekg(0, std::ios::end);
        const size_t size = static_cast<size_t>(stream.tellg());
        stream.seekg(0, std::ios::beg);
        std::vector<char> bytes(size);
        stream.read(bytes.data(), static_cast<std::streamsize>(size));

        runtime_.reset(nvinfer1::createInferRuntime(g_logger));
        if (!runtime_) throw std::runtime_error("createInferRuntime failed");
        engine_.reset(runtime_->deserializeCudaEngine(bytes.data(), bytes.size()));
        if (!engine_) throw std::runtime_error("deserializeCudaEngine failed");
        context_.reset(engine_->createExecutionContext());
        if (!context_) throw std::runtime_error("createExecutionContext failed");
        check_cuda(cudaStreamCreate(&cuda_stream_), "cudaStreamCreate");

        const int tensor_count = engine_->getNbIOTensors();
        int input_count = 0;
        int output_count = 0;
        for (int index = 0; index < tensor_count; ++index) {
            TensorBinding binding;
            binding.name = engine_->getIOTensorName(index);
            binding.is_input = engine_->getTensorIOMode(binding.name.c_str()) ==
                               nvinfer1::TensorIOMode::kINPUT;
            binding.dtype = engine_->getTensorDataType(binding.name.c_str());
            binding.dims = engine_->getTensorShape(binding.name.c_str());
            binding.elements = volume(binding.dims);
            binding.bytes = binding.elements * dtype_size(binding.dtype);
            check_cuda(cudaMalloc(&binding.device, binding.bytes), "cudaMalloc");
            if (binding.is_input) {
                ++input_count;
                check_cuda(cudaMallocHost(&binding.input_host, binding.bytes), "cudaMallocHost input");
                input_index_ = static_cast<int>(bindings_.size());
            } else {
                ++output_count;
            }
            if (!context_->setTensorAddress(binding.name.c_str(), binding.device)) {
                throw std::runtime_error("setTensorAddress failed for " + binding.name);
            }
            bindings_.push_back(binding);
        }
        if (input_count != 1 || output_count != 3 || input_index_ < 0) {
            std::ostringstream oss;
            oss << "three-stage canary requires exactly one TensorRT input and three raw-head outputs: inputs="
                << input_count << " outputs=" << output_count;
            throw std::runtime_error(oss.str());
        }
        identify_heads();
        allocate_post_slots();
        std::cerr << "[three-stage][trt] input=" << input().name
                  << " dtype=" << dtype_name(input().dtype)
                  << " bytes=" << input().bytes
                  << " outputs=" << output_description() << std::endl;
    }

    ~TRTStage() {
        for (auto &slot : post_slots_) {
            if (slot.head8) cudaFreeHost(slot.head8);
            if (slot.head16) cudaFreeHost(slot.head16);
            if (slot.head32) cudaFreeHost(slot.head32);
        }
        for (auto &binding : bindings_) {
            if (binding.device) cudaFree(binding.device);
            if (binding.input_host) cudaFreeHost(binding.input_host);
        }
        if (cuda_stream_) cudaStreamDestroy(cuda_stream_);
    }

    const TensorBinding &input() const {
        return bindings_.at(static_cast<size_t>(input_index_));
    }

    const PostSlot &post_slot(int index) const {
        return post_slots_.at(static_cast<size_t>(index));
    }

    size_t head8_elements() const { return bindings_.at(static_cast<size_t>(head8_index_)).elements; }
    size_t head16_elements() const { return bindings_.at(static_cast<size_t>(head16_index_)).elements; }
    size_t head32_elements() const { return bindings_.at(static_cast<size_t>(head32_index_)).elements; }

    std::string output_description() const {
        std::ostringstream out;
        bool first = true;
        for (int index : {head8_index_, head16_index_, head32_index_}) {
            const auto &binding = bindings_.at(static_cast<size_t>(index));
            if (!first) out << ",";
            first = false;
            out << binding.name << ":";
            for (int dim = 0; dim < binding.dims.nbDims; ++dim) {
                if (dim) out << "x";
                out << binding.dims.d[dim];
            }
        }
        return out.str();
    }

    void copy_input_from_boundary(const std::vector<uint8_t> &boundary) {
        TensorBinding &binding = bindings_.at(static_cast<size_t>(input_index_));
        if (binding.dtype != nvinfer1::DataType::kUINT8 &&
            binding.dtype != nvinfer1::DataType::kINT8) {
            throw std::runtime_error("three-stage canary requires UINT8/INT8 TensorRT boundary input");
        }
        if (boundary.size() != binding.bytes) {
            std::ostringstream oss;
            oss << "boundary size=" << boundary.size()
                << " does not match TensorRT input bytes=" << binding.bytes;
            throw std::runtime_error(oss.str());
        }
        std::memcpy(binding.input_host, boundary.data(), binding.bytes);
    }

    void run_to_post_slot(int slot_index) {
        TensorBinding &input_binding = bindings_.at(static_cast<size_t>(input_index_));
        check_cuda(cudaMemcpyAsync(input_binding.device, input_binding.input_host,
                                   input_binding.bytes, cudaMemcpyHostToDevice, cuda_stream_),
                   "TensorRT input H2D");
        if (!context_->enqueueV3(cuda_stream_)) {
            throw std::runtime_error("TensorRT enqueueV3 failed");
        }
        const PostSlot &slot = post_slots_.at(static_cast<size_t>(slot_index));
        copy_output_to_host(head8_index_, slot.head8);
        copy_output_to_host(head16_index_, slot.head16);
        copy_output_to_host(head32_index_, slot.head32);
        check_cuda(cudaStreamSynchronize(cuda_stream_), "cudaStreamSynchronize");
    }

private:
    struct RuntimeDeleter { void operator()(nvinfer1::IRuntime *p) const { delete p; } };
    struct EngineDeleter { void operator()(nvinfer1::ICudaEngine *p) const { delete p; } };
    struct ContextDeleter { void operator()(nvinfer1::IExecutionContext *p) const { delete p; } };

    void copy_output_to_host(int binding_index, void *host) {
        const auto &binding = bindings_.at(static_cast<size_t>(binding_index));
        check_cuda(cudaMemcpyAsync(host, binding.device, binding.bytes,
                                   cudaMemcpyDeviceToHost, cuda_stream_),
                   "TensorRT output D2H");
    }

    static int grid_size_from_dims(const nvinfer1::Dims &dims) {
        if (dims.nbDims != 5) return -1;
        if (dims.d[0] != 1 || dims.d[1] != 3 || dims.d[4] != 85) return -1;
        if (dims.d[2] != dims.d[3]) return -1;
        return dims.d[2];
    }

    void identify_heads() {
        for (size_t index = 0; index < bindings_.size(); ++index) {
            const auto &binding = bindings_[index];
            if (binding.is_input) continue;
            if (binding.dtype != nvinfer1::DataType::kFLOAT) {
                throw std::runtime_error("YOLOv7 raw heads must be float32");
            }
            const int grid = grid_size_from_dims(binding.dims);
            if (grid == 80) head8_index_ = static_cast<int>(index);
            else if (grid == 40) head16_index_ = static_cast<int>(index);
            else if (grid == 20) head32_index_ = static_cast<int>(index);
        }
        if (head8_index_ < 0 || head16_index_ < 0 || head32_index_ < 0) {
            throw std::runtime_error("expected YOLOv7 80/40/20 raw-head output tensors");
        }
        const size_t expected8 = static_cast<size_t>(1) * 3 * 80 * 80 * 85;
        const size_t expected16 = static_cast<size_t>(1) * 3 * 40 * 40 * 85;
        const size_t expected32 = static_cast<size_t>(1) * 3 * 20 * 20 * 85;
        if (head8_elements() != expected8 || head16_elements() != expected16 ||
            head32_elements() != expected32) {
            throw std::runtime_error("YOLOv7 raw-head element count mismatch");
        }
    }

    void allocate_post_slots() {
        const size_t bytes8 = head8_elements() * sizeof(float);
        const size_t bytes16 = head16_elements() * sizeof(float);
        const size_t bytes32 = head32_elements() * sizeof(float);
        for (auto &slot : post_slots_) {
            check_cuda(cudaMallocHost(reinterpret_cast<void **>(&slot.head8), bytes8),
                       "cudaMallocHost head8");
            check_cuda(cudaMallocHost(reinterpret_cast<void **>(&slot.head16), bytes16),
                       "cudaMallocHost head16");
            check_cuda(cudaMallocHost(reinterpret_cast<void **>(&slot.head32), bytes32),
                       "cudaMallocHost head32");
        }
    }

    std::unique_ptr<nvinfer1::IRuntime, RuntimeDeleter> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, EngineDeleter> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, ContextDeleter> context_;
    std::vector<TensorBinding> bindings_;
    std::vector<PostSlot> post_slots_;
    int input_index_ = -1;
    int head8_index_ = -1;
    int head16_index_ = -1;
    int head32_index_ = -1;
    cudaStream_t cuda_stream_ = nullptr;
};

struct BoundarySlot {
    int sequence = 0;
    int image_index = 0;
    int original_width = 0;
    int original_height = 0;
    std::vector<uint8_t> boundary;
    double preprocess_ms = 0.0;
    double p1_ms = 0.0;
};

struct PostWorkItem {
    int sequence = 0;
    int image_index = 0;
    int original_width = 0;
    int original_height = 0;
    int post_slot = 0;
};

struct FrameMetrics {
    double preprocess_ms = 0.0;
    double p1_ms = 0.0;
    double handoff_ms = 0.0;
    double p2_ms = 0.0;
    double post_ms = 0.0;
    double boundary_wait_ms = 0.0;
    double post_slot_wait_ms = 0.0;
    int detection_count = 0;
    int callback_status = 0;
};

struct PhaseResult {
    int frames_requested = 0;
    int raw_frames_completed = 0;
    int completed_frames = 0;
    int callback_failures = 0;
    double raw_makespan_ms = 0.0;
    double completed_makespan_ms = 0.0;
    std::vector<FrameMetrics> frames;
};

static PhaseResult run_phase(
    int frame_count,
    bool measured,
    const std::vector<std::string> &images,
    int p1_queue_depth,
    int post_queue_depth,
    int pad_value,
    HailoStage &hailo,
    TRTStage &trt,
    ThreeStagePostCallbackV1 callback,
    void *user_data) {

    PhaseResult result;
    result.frames_requested = frame_count;
    result.frames.resize(static_cast<size_t>(frame_count));
    if (frame_count <= 0) return result;

    std::vector<BoundarySlot> boundary_slots(static_cast<size_t>(p1_queue_depth));
    for (auto &slot : boundary_slots) slot.boundary.resize(hailo.output_frame_size());

    BlockingQueue<int> free_boundary(static_cast<size_t>(p1_queue_depth));
    BlockingQueue<int> p1_to_p2(static_cast<size_t>(p1_queue_depth));
    BlockingQueue<int> free_post(static_cast<size_t>(post_queue_depth));
    BlockingQueue<PostWorkItem> p2_to_post(static_cast<size_t>(post_queue_depth));
    for (int index = 0; index < p1_queue_depth; ++index) free_boundary.push(index);
    for (int index = 0; index < post_queue_depth; ++index) free_post.push(index);

    std::exception_ptr thread_error;
    std::mutex error_mutex;
    auto capture_error = [&](std::exception_ptr value) {
        std::lock_guard<std::mutex> lock(error_mutex);
        if (!thread_error) thread_error = value;
        free_boundary.close();
        p1_to_p2.close();
        free_post.close();
        p2_to_post.close();
    };

    Clock::time_point measurement_start{};
    Clock::time_point raw_end{};
    Clock::time_point completed_end{};
    std::atomic<int> raw_count{0};
    std::atomic<int> completed_count{0};
    std::atomic<int> callback_failures{0};

    std::thread p1_thread([&] {
        try {
            for (int sequence = 0; sequence < frame_count; ++sequence) {
                const auto wait_start = Clock::now();
                int slot_index = -1;
                if (!free_boundary.pop(slot_index)) break;
                const auto wait_end = Clock::now();
                auto &metrics = result.frames.at(static_cast<size_t>(sequence));
                metrics.boundary_wait_ms = ms_between(wait_start, wait_end);

                auto &slot = boundary_slots.at(static_cast<size_t>(slot_index));
                slot.sequence = sequence;
                slot.image_index = sequence % static_cast<int>(images.size());
                const auto stage_start = Clock::now();
                if (measured && sequence == 0) measurement_start = stage_start;
                cv::Mat bgr = cv::imread(images.at(static_cast<size_t>(slot.image_index)));
                if (bgr.empty()) throw std::runtime_error("failed to read image");
                slot.original_width = bgr.cols;
                slot.original_height = bgr.rows;
                cv::Mat rgb = letterbox_rgb_uint8(
                    bgr, hailo.input_width(), hailo.input_height(), pad_value);
                const auto preprocess_end = Clock::now();
                hailo.infer(rgb, slot.boundary);
                const auto p1_end = Clock::now();
                slot.preprocess_ms = ms_between(stage_start, preprocess_end);
                slot.p1_ms = ms_between(preprocess_end, p1_end);
                p1_to_p2.push(slot_index);
            }
            p1_to_p2.close();
        } catch (...) {
            capture_error(std::current_exception());
        }
    });

    std::thread p2_thread([&] {
        try {
            int boundary_index = -1;
            while (p1_to_p2.pop(boundary_index)) {
                auto &slot = boundary_slots.at(static_cast<size_t>(boundary_index));
                const int sequence = slot.sequence;
                auto &metrics = result.frames.at(static_cast<size_t>(sequence));
                metrics.preprocess_ms = slot.preprocess_ms;
                metrics.p1_ms = slot.p1_ms;

                const auto post_wait_start = Clock::now();
                int post_slot_index = -1;
                if (!free_post.pop(post_slot_index)) break;
                const auto post_wait_end = Clock::now();
                metrics.post_slot_wait_ms = ms_between(post_wait_start, post_wait_end);

                const auto handoff_start = Clock::now();
                trt.copy_input_from_boundary(slot.boundary);
                const auto handoff_end = Clock::now();
                trt.run_to_post_slot(post_slot_index);
                const auto p2_end = Clock::now();
                metrics.handoff_ms = ms_between(handoff_start, handoff_end);
                metrics.p2_ms = ms_between(handoff_end, p2_end);
                raw_count.fetch_add(1);
                if (measured) raw_end = p2_end;

                PostWorkItem work;
                work.sequence = sequence;
                work.image_index = slot.image_index;
                work.original_width = slot.original_width;
                work.original_height = slot.original_height;
                work.post_slot = post_slot_index;
                p2_to_post.push(work);
                free_boundary.push(boundary_index);
            }
            p2_to_post.close();
        } catch (...) {
            capture_error(std::current_exception());
        }
    });

    std::thread post_thread([&] {
        try {
            PostWorkItem work;
            while (p2_to_post.pop(work)) {
                const auto &slot = trt.post_slot(work.post_slot);
                int32_t detection_count = 0;
                const auto post_start = Clock::now();
                const int status = callback(
                    measured ? 1 : 0,
                    work.sequence,
                    work.image_index,
                    work.original_width,
                    work.original_height,
                    slot.head8,
                    static_cast<uint64_t>(trt.head8_elements()),
                    slot.head16,
                    static_cast<uint64_t>(trt.head16_elements()),
                    slot.head32,
                    static_cast<uint64_t>(trt.head32_elements()),
                    &detection_count,
                    user_data);
                const auto post_end = Clock::now();
                auto &metrics = result.frames.at(static_cast<size_t>(work.sequence));
                metrics.post_ms = ms_between(post_start, post_end);
                metrics.detection_count = detection_count;
                metrics.callback_status = status;
                if (status != 0) callback_failures.fetch_add(1);
                else completed_count.fetch_add(1);
                if (measured) completed_end = post_end;
                free_post.push(work.post_slot);
            }
        } catch (...) {
            capture_error(std::current_exception());
        }
    });

    p1_thread.join();
    p2_thread.join();
    post_thread.join();

    if (thread_error) std::rethrow_exception(thread_error);

    result.raw_frames_completed = raw_count.load();
    result.completed_frames = completed_count.load();
    result.callback_failures = callback_failures.load();
    if (measured) {
        result.raw_makespan_ms = ms_between(measurement_start, raw_end);
        result.completed_makespan_ms = ms_between(measurement_start, completed_end);
    }
    return result;
}

static void write_report(
    const ThreeStageConfigV1 &config,
    const std::vector<std::string> &images,
    const TRTStage &trt,
    const PhaseResult &warmup,
    const PhaseResult &measurement,
    const std::string &out_path) {

    std::vector<double> preprocess;
    std::vector<double> p1;
    std::vector<double> p1_stage;
    std::vector<double> handoff;
    std::vector<double> p2;
    std::vector<double> p2_stage;
    std::vector<double> post;
    std::vector<double> boundary_wait;
    std::vector<double> post_slot_wait;
    std::vector<double> detection_counts;
    preprocess.reserve(measurement.frames.size());
    p1.reserve(measurement.frames.size());
    p1_stage.reserve(measurement.frames.size());
    handoff.reserve(measurement.frames.size());
    p2.reserve(measurement.frames.size());
    p2_stage.reserve(measurement.frames.size());
    post.reserve(measurement.frames.size());
    boundary_wait.reserve(measurement.frames.size());
    post_slot_wait.reserve(measurement.frames.size());
    detection_counts.reserve(measurement.frames.size());

    for (const auto &frame : measurement.frames) {
        preprocess.push_back(frame.preprocess_ms);
        p1.push_back(frame.p1_ms);
        p1_stage.push_back(frame.preprocess_ms + frame.p1_ms);
        handoff.push_back(frame.handoff_ms);
        p2.push_back(frame.p2_ms);
        p2_stage.push_back(frame.handoff_ms + frame.p2_ms);
        post.push_back(frame.post_ms);
        boundary_wait.push_back(frame.boundary_wait_ms);
        post_slot_wait.push_back(frame.post_slot_wait_ms);
        detection_counts.push_back(static_cast<double>(frame.detection_count));
    }

    const Summary preprocess_summary = summarize(preprocess);
    const Summary p1_summary = summarize(p1);
    const Summary p1_stage_summary = summarize(p1_stage);
    const Summary handoff_summary = summarize(handoff);
    const Summary p2_summary = summarize(p2);
    const Summary p2_stage_summary = summarize(p2_stage);
    const Summary post_summary = summarize(post);
    const Summary boundary_wait_summary = summarize(boundary_wait);
    const Summary post_slot_wait_summary = summarize(post_slot_wait);
    const Summary detection_summary = summarize(detection_counts);

    const double raw_fps = measurement.raw_makespan_ms > 0.0
        ? 1000.0 * static_cast<double>(measurement.raw_frames_completed) /
          measurement.raw_makespan_ms
        : 0.0;
    const double completed_fps = measurement.completed_makespan_ms > 0.0
        ? 1000.0 * static_cast<double>(measurement.completed_frames) /
          measurement.completed_makespan_ms
        : 0.0;
    const double theoretical_raw_cycle_ms = std::max(
        p1_stage_summary.mean, p2_stage_summary.mean);
    const double theoretical_raw_fps = theoretical_raw_cycle_ms > 0.0
        ? 1000.0 / theoretical_raw_cycle_ms : 0.0;
    const double theoretical_cycle_ms = std::max({
        p1_stage_summary.mean, p2_stage_summary.mean, post_summary.mean});
    const double theoretical_fps = theoretical_cycle_ms > 0.0
        ? 1000.0 / theoretical_cycle_ms : 0.0;

    fs::create_directories(fs::path(out_path).parent_path());
    std::ofstream out(out_path);
    if (!out) throw std::runtime_error("failed to create report: " + out_path);
    out << std::fixed << std::setprecision(9);
    out << "{\n";
    out << "  \"schema\": \"onnx-splitpoint/yolov7-native-three-stage-runtime\",\n";
    out << "  \"schema_version\": 1,\n";
    out << "  \"ok\": " << ((measurement.callback_failures == 0 &&
                                      measurement.raw_frames_completed == config.frames &&
                                      measurement.completed_frames == config.frames) ? "true" : "false") << ",\n";
    out << "  \"mode\": \"hailo8_cpp_fifo_trt_ctypes_fast_numpy_three_stage\",\n";
    out << "  \"measurement_endpoint_raw\": \"raw_model_outputs\",\n";
    out << "  \"measurement_endpoint_completed\": \"completed_detection\",\n";
    out << "  \"frames\": " << config.frames << ",\n";
    out << "  \"warmup\": " << config.warmup << ",\n";
    out << "  \"warmup_fully_drained_before_measurement\": true,\n";
    out << "  \"image_count\": " << images.size() << ",\n";
    out << "  \"p1_queue_depth\": " << config.p1_queue_depth << ",\n";
    out << "  \"post_queue_depth\": " << config.post_queue_depth << ",\n";
    out << "  \"callback_abi\": \"ctypes_zero_copy_pinned_host_raw_heads_v1\",\n";
    out << "  \"runtime_artifacts\": {\"hef\":\"" << json_escape(config.hef_path)
        << "\",\"engine\":\"" << json_escape(config.engine_path) << "\"},\n";
    out << "  \"output_contract\": {\"head_stride8_elements\":" << trt.head8_elements()
        << ",\"head_stride16_elements\":" << trt.head16_elements()
        << ",\"head_stride32_elements\":" << trt.head32_elements()
        << ",\"dtype\":\"float32\"},\n";
    out << "  \"warmup_result\": {\"requested\":" << warmup.frames_requested
        << ",\"raw_completed\":" << warmup.raw_frames_completed
        << ",\"completed\":" << warmup.completed_frames
        << ",\"callback_failures\":" << warmup.callback_failures << "},\n";
    out << "  \"raw_model_outputs\": {\"completed_frames\":"
        << measurement.raw_frames_completed << ",\"makespan_ms\":"
        << measurement.raw_makespan_ms << ",\"throughput_fps\":" << raw_fps << "},\n";
    out << "  \"completed_detection\": {\"completed_frames\":"
        << measurement.completed_frames << ",\"makespan_ms\":"
        << measurement.completed_makespan_ms << ",\"throughput_fps\":"
        << completed_fps << ",\"raw_ratio\":"
        << (raw_fps > 0.0 ? completed_fps / raw_fps : 0.0) << "},\n";
    out << "  \"stage_metrics\": {\n";
    out << "    \"P1\": {\"preprocess\":"; write_summary(out, preprocess_summary);
    out << ",\"inference\":"; write_summary(out, p1_summary);
    out << ",\"stage\":"; write_summary(out, p1_stage_summary); out << "},\n";
    out << "    \"P2\": {\"handoff\":"; write_summary(out, handoff_summary);
    out << ",\"inference_and_raw_output_d2h\":"; write_summary(out, p2_summary);
    out << ",\"stage\":"; write_summary(out, p2_stage_summary); out << "},\n";
    out << "    \"postprocessing\": {\"fast_decode_nms_inverse_letterbox_records\":";
    write_summary(out, post_summary); out << "}\n";
    out << "  },\n";
    out << "  \"queue_waits\": {\"P1_free_boundary\":";
    write_summary(out, boundary_wait_summary);
    out << ",\"P2_free_post_slot\":"; write_summary(out, post_slot_wait_summary);
    out << "},\n";
    out << "  \"detection_count\":"; write_numeric_summary(out, detection_summary); out << ",\n";
    out << "  \"theoretical_raw_cycle_ms\": " << theoretical_raw_cycle_ms << ",\n";
    out << "  \"theoretical_raw_fps\": " << theoretical_raw_fps << ",\n";
    out << "  \"theoretical_three_stage_cycle_ms\": " << theoretical_cycle_ms << ",\n";
    out << "  \"theoretical_three_stage_fps\": " << theoretical_fps << ",\n";
    out << "  \"callback_failures\": " << measurement.callback_failures << "\n";
    out << "}\n";
}

extern "C" size_t onnx_splitpoint_three_stage_config_size_v1() {
    return sizeof(ThreeStageConfigV1);
}

extern "C" int onnx_splitpoint_run_three_stage_v1(
    const ThreeStageConfigV1 *config,
    ThreeStagePostCallbackV1 callback,
    void *user_data,
    char *error_buffer,
    size_t error_buffer_size) {

    try {
        if (!config) throw std::runtime_error("config is null");
        if (config->abi_version != 1) throw std::runtime_error("unsupported ABI version");
        if (config->struct_size != sizeof(ThreeStageConfigV1)) {
            std::ostringstream oss;
            oss << "config struct size mismatch: got=" << config->struct_size
                << " expected=" << sizeof(ThreeStageConfigV1);
            throw std::runtime_error(oss.str());
        }
        if (!callback) throw std::runtime_error("postprocess callback is null");
        if (!config->hef_path || !config->engine_path || !config->images_dir || !config->out_json) {
            throw std::runtime_error("required config path is null");
        }
        if (config->frames <= 0 || config->warmup < 0 || config->p1_queue_depth <= 0 ||
            config->post_queue_depth <= 0) {
            throw std::runtime_error("invalid frame, warmup or queue setting");
        }

        const std::vector<std::string> images = list_images(config->images_dir);
        HailoStage hailo(config->hef_path, config->device_id ? config->device_id : "");
        TRTStage trt(config->engine_path, config->post_queue_depth);

        const PhaseResult warmup = run_phase(
            config->warmup, false, images, config->p1_queue_depth,
            config->post_queue_depth, config->letterbox_pad_value,
            hailo, trt, callback, user_data);
        if (warmup.callback_failures != 0 || warmup.completed_frames != config->warmup) {
            throw std::runtime_error("warmup phase did not complete cleanly");
        }
        const PhaseResult measurement = run_phase(
            config->frames, true, images, config->p1_queue_depth,
            config->post_queue_depth, config->letterbox_pad_value,
            hailo, trt, callback, user_data);

        write_report(*config, images, trt, warmup, measurement, config->out_json);
        set_error(error_buffer, error_buffer_size, "");
        return (measurement.callback_failures == 0 &&
                measurement.raw_frames_completed == config->frames &&
                measurement.completed_frames == config->frames) ? 0 : 2;
    } catch (const std::exception &error) {
        set_error(error_buffer, error_buffer_size, error.what());
        return 1;
    } catch (...) {
        set_error(error_buffer, error_buffer_size, "unknown C++ exception");
        return 1;
    }
}
