"""Actual C++ scheduling/C ABI and normal completion decoder; SDK leaves only fake.

No HEF/engine compilation, accelerator access or model inference occurs. The
software library extracts the unchanged non-SDK C++ functions and replaces only
OpenCV/Hailo/TRT interfaces. Build products and results belong to pytest tmp_path.
"""
from __future__ import annotations

import copy
import ctypes
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RESOURCE = ROOT / "onnx_splitpoint_tool/resources/native_concurrent_three_stage_yolov7"


def load_canary(monkeypatch):
    monkeypatch.syspath_prepend(str(RESOURCE))
    spec = importlib.util.spec_from_file_location("campaign_runtime_test", RESOURCE / "three_stage_canary.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SDK_TYPES = r'''
#define CV_8UC3 0
namespace nvinfer1 {
enum class DataType { kFLOAT, kHALF, kINT8, kUINT8, kINT32, kBOOL };
struct Dims { int nbDims=0; int d[8]{}; };
}
namespace cv {
struct Mat {
 int rows=0, cols=0; std::shared_ptr<std::vector<uint8_t>> data;
 Mat()=default;
 Mat(int h,int w,int):rows(h),cols(w),data(std::make_shared<std::vector<uint8_t>>(h*w*3)){}
 bool empty() const { return !data; }
 size_t total() const { return size_t(rows)*cols; }
 size_t elemSize() const { return 3; }
 template<class T> T* ptr(int) { return reinterpret_cast<T*>(data->data()); }
 template<class T> const T* ptr(int) const { return reinterpret_cast<const T*>(data->data()); }
};
static Mat imread(const std::string&) { throw std::runtime_error("UNEXPECTED_IMAGE_DECODE"); }
}
static cv::Mat letterbox_rgb_uint8(const cv::Mat&,int,int,int) { throw std::runtime_error("UNEXPECTED_LETTERBOX"); }
static std::atomic<int> active_sdks{0}, max_active_sdks{0}, hailo_calls{0}, trt_calls{0};
struct Active {
 Active() { int n=++active_sdks; int old=max_active_sdks.load(); while(n>old && !max_active_sdks.compare_exchange_weak(old,n)) {} }
 ~Active() { --active_sdks; }
};
struct FakeInfo { std::string name; struct { int height,width,features; } shape; };
class HailoStage {
public:
 HailoStage(const std::string& path, const std::string&) { if(path.find("wrong")!=std::string::npos) throw std::runtime_error("wrong HEF"); }
 int input_width() const { return 640; } int input_height() const { return 640; }
 size_t output_frame_size() const { return 80*80*512; }
 FakeInfo input_info() const { return {"images",{640,640,3}}; }
 FakeInfo output_info() const { return {"boundary",{80,80,512}}; }
 void infer(const cv::Mat& input, std::vector<uint8_t>& output) {
  Active active; ++hailo_calls; output[0]=input.ptr<uint8_t>(0)[0];
  std::this_thread::sleep_for(std::chrono::microseconds(80));
 }
};
'''
SDK_TRT = r'''
class TRTStage {
 std::vector<TensorBinding> b; std::vector<PostSlot> slots;
 std::vector<std::vector<float>> payload;
public:
 TRTStage(const std::string& path,int depth) {
  if(path.find("wrong")!=std::string::npos) throw std::runtime_error("wrong engine");
  TensorBinding in; in.name="feed"; in.dtype=nvinfer1::DataType::kUINT8; in.dims.nbDims=4;
  in.dims.d[0]=1;in.dims.d[1]=512;in.dims.d[2]=80;in.dims.d[3]=80; b.push_back(in);
  int i=0;for(int grid:{80,40,20}) { TensorBinding t; t.name=i==0?"output":i==1?"clone_1":"clone_2";
   t.dtype=nvinfer1::DataType::kFLOAT;t.dims.nbDims=5;
   t.dims.d[0]=1;t.dims.d[1]=3;t.dims.d[2]=grid;t.dims.d[3]=grid;t.dims.d[4]=85;b.push_back(t);++i; }
  payload.reserve(depth*3);slots.resize(depth);
  for(int d=0;d<depth;++d) {for(int grid:{80,40,20}) {
   payload.emplace_back(3*grid*grid*85,-100.f);
   if(grid==80) {auto &v=payload.back();v[0]=0;v[1]=0;v[2]=0;v[3]=0;v[4]=10;v[5]=10;}
  } slots[d]={payload[d*3].data(),payload[d*3+1].data(),payload[d*3+2].data()}; }
 }
 const TensorBinding& input() const { return b[0]; }
 const TensorBinding& head_binding(int grid) const { return b[grid==80?1:grid==40?2:3]; }
 size_t head8_elements() const {return 3*80*80*85;} size_t head16_elements() const {return 3*40*40*85;}
 size_t head32_elements() const {return 3*20*20*85;}
 const PostSlot& post_slot(int i) const {return slots.at(i);}
 void copy_input_from_boundary(const std::vector<uint8_t>&) {}
 void run_to_post_slot(int) { Active active; ++trt_calls; std::this_thread::sleep_for(std::chrono::microseconds(80)); }
};
'''


def compile_software_runtime(tmp: Path, *, boundary_contract=None, input_name="images") -> Path:
    """Reusable SDK-only library factory for Parent's connected writer checks."""
    tmp.mkdir(parents=True, exist_ok=True)
    cpp = (RESOURCE / "cpp/three_stage_canary.cpp").read_text()
    def part(start, end):
        return cpp[cpp.index(start):cpp.index(end)]
    sdk_types, sdk_trt = SDK_TYPES, SDK_TRT
    if boundary_contract is not None:
        h,w,c = boundary_contract["shape"]
        sdk_types = sdk_types.replace('return 80*80*512;', f'return {h}*{w}*{c};')
        sdk_types = sdk_types.replace('{"boundary",{80,80,512}}',
            '{' + json.dumps(boundary_contract["name"]) + ',{' + f'{h},{w},{c}' + '}}')
        sdk_trt = sdk_trt.replace('in.name="feed";', 'in.name=' + json.dumps(boundary_contract["trt_input"]["name"]) + ';')
        sdk_trt = sdk_trt.replace('in.dims.d[1]=512;in.dims.d[2]=80;in.dims.d[3]=80;',
            f'in.dims.d[1]={c};in.dims.d[2]={h};in.dims.d[3]={w};')
    sdk_types = sdk_types.replace('return {"images",{640,640,3}}', 'return {' + json.dumps(input_name) + ',{640,640,3}}')
    source = (
        part("#include <algorithm>", "static inline void check_cuda") + sdk_types
        + part("static std::string dtype_name", "static cv::Mat letterbox_rgb_uint8")
        + part("struct TensorBinding", "class TRTStage") + sdk_trt
        + cpp[cpp.index("struct FrameMetrics"):]
        + '\nextern "C" int fake_hailo_calls(){return hailo_calls;}\n'
        + 'extern "C" int fake_trt_calls(){return trt_calls;}\n'
        + 'extern "C" int fake_max_concurrency(){return max_active_sdks;}\n'
    )
    path = tmp / "software_runtime.cpp"
    path.write_text(source)
    library = tmp / "software_runtime.so"
    completed = subprocess.run(["g++", "-std=c++17", "-O2", "-shared", "-fPIC", "-pthread", str(path), "-o", str(library)],
                               text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (tmp / "compile.log").write_text(completed.stdout)
    assert completed.returncode == 0, completed.stdout
    return library


def campaign_config(tmp: Path, canary):
    tmp.mkdir(parents=True, exist_ok=True)
    contract = json.loads((RESOURCE / "reference/current_completed_task_result.json").read_text())["completion_execution_contract"]
    def artifact(name, contents):
        path = tmp / name
        path.write_bytes(contents)
        return {"path": str(path), "sha256": canary.sha256_file(path), "size_bytes": path.stat().st_size}
    prepared = artifact("prepared.bin", np.zeros((640,640,3), dtype=np.uint8).tobytes())
    return {"hef": artifact("bound.hef", b"software SDK HEF"), "engine": artifact("bound.engine", b"software SDK current engine"),
        "input_image": artifact("bound.jpg", b"software image (not decoded)"),
        "prepared_input": {**prepared, "name":"images", "shape":[640,640,3], "dtype":"uint8", "format":"raw"},
        "original_wh": contract["processor_contract"]["original_wh"],
        "completion_contract": contract, "processor_contract":contract["processor_contract"],
        "outputs":[{key:s[key] for key in ("name","shape","dtype")} for s in contract["source_endpoint"]["tensor_signature"]["tensors"]],
        "boundary_contract":{"name":"boundary","shape":[80,80,512],"dtype":"uint8",
                             "trt_input":{"name":"feed","shape":[1,512,80,80],"dtype":"uint8"}},
        "frames":1000,"warmup":100,"repetitions":3,"duration_s":0.0,"p1_queue_depth":3,"post_queue_depth":4,
        "device_id":"", "tool_root":str(ROOT)}


@pytest.fixture(scope="module")
def sdk_library(tmp_path_factory):
    return compile_software_runtime(tmp_path_factory.mktemp("sdk-library"))


@pytest.fixture
def canary(monkeypatch):
    return load_canary(monkeypatch)


def test_real_cpp_three_repetitions_and_normal_fast_completion(canary, sdk_library, monkeypatch, tmp_path):
    cfg = campaign_config(tmp_path / "inputs", canary)
    monkeypatch.setattr(canary, "build_cpp", lambda _: sdk_library)
    monkeypatch.setattr(canary, "choose_exact_artifact", lambda _: pytest.fail("historical artifact lookup"))
    report = canary.run_campaign(cfg, out_root=tmp_path / "output")
    assert report["ok"] is True and report["config"] == cfg
    assert len(report["repetitions"]) == 3
    assert len({r["runtime_instance_id"] for r in report["repetitions"]}) == 3
    library = ctypes.CDLL(str(sdk_library))
    assert library.fake_hailo_calls() >= 3300 and library.fake_trt_calls() >= 3300
    assert library.fake_max_concurrency() >= 2
    for row in report["repetitions"]:
        assert row["frames"] == row["completed_work_units"] == row["produced_frames"] == 1000
        assert row["warmup_completed_frames"] == 100
        assert row["completion_attestation"]["completed_work_units"] == 1000
        assert row["completion_attestation"]["status"] == "passed"
        assert row["measurement_start_ns"] < row["measurement_end_ns"] <= row["drain_end_ns"]
        assert row["preprocessing_included"] is False
        assert np.load(row["boundary_array"]["path"]).shape == (1,512,80,80)
        assert set(row["output_arrays"]) == {"output", "clone_1", "clone_2"}
        cpp = json.loads(Path(row["runtime_report"]).read_text())
        assert cpp["runtime_artifacts"] == {key:cfg[key]["path"] for key in ("hef","engine")}
        assert cpp["stage_metrics"]["P1"]["preprocess"]["mean_ms"] == 0


@pytest.mark.parametrize("damage", ["engine", "input", "frames", "warmup", "repetitions", "duration", "output", "boundary", "processor"])
def test_bound_campaign_rejects_damage_before_build(canary, monkeypatch, tmp_path, damage):
    cfg = campaign_config(tmp_path / "inputs", canary)
    if damage in ("engine", "input"):
        cfg["engine" if damage=="engine" else "prepared_input"]["sha256"] = "0"*64
    elif damage in ("frames","warmup","repetitions"):
        cfg[damage] -= 1
    elif damage == "duration": cfg["duration_s"] = 2
    elif damage == "output": cfg["outputs"][0]["name"] = "foreign"
    elif damage == "boundary": cfg["boundary_contract"]["trt_input"]["shape"] = [1,80,80,512]
    else: cfg["processor_contract"] = {**cfg["processor_contract"],"max_detections":1}
    monkeypatch.setattr(canary, "build_cpp", lambda _: pytest.fail("build reached"))
    with pytest.raises(RuntimeError): canary.run_campaign(cfg, out_root=tmp_path / "output")


def test_actual_cpp_runtime_interface_rejects_bound_name_mismatch(canary,sdk_library,tmp_path):
    cfg=campaign_config(tmp_path / "inputs",canary)
    cfg["boundary_contract"]["name"]="foreign"
    context=canary.CampaignCallbackContext(cfg,{})
    out=tmp_path / "runtime.json"
    with pytest.raises(RuntimeError,match="bound runtime interface mismatch"):
        canary.NativeLibrary(sdk_library).run_campaign(cfg,out_json=out,context=context,runtime_instance_id="fixture")
    assert context.measured.completed_count == context.warmup.completed_count == 0


def test_actual_cpp_duration_counts_are_time_based_and_no_tensor_dumps(canary,sdk_library,tmp_path):
    cfg=campaign_config(tmp_path / "inputs",canary)
    # Short unit check of the very same duration loop. Public energy admission
    # still requires 60 seconds; no collector is run in this software test.
    cfg.update(duration_s=0.12,frames=1,warmup=0,repetitions=1)
    context=canary.CampaignCallbackContext(cfg,{"runtime_instance_id":"duration-test"})
    out=tmp_path / "runtime.json"
    runtime=canary.NativeLibrary(sdk_library).run_campaign(cfg,out_json=out,context=context,
        runtime_instance_id="duration-test",persist_sample=False)
    assert runtime["completed_detection"]["completed_frames"] > 1
    assert runtime["completed_detection"]["makespan_ms"] >= 120
    assert runtime["sample_saved_after_measurement_and_drain"] is False
    assert not list(tmp_path.glob("last_*.bin"))
    assert context.measured.attestation()["status"] == "passed"


def test_current_callback_failure_remains_failed_and_drains(canary,sdk_library,tmp_path):
    cfg=campaign_config(tmp_path / "inputs",canary)
    cfg.update(frames=4,warmup=0)
    context=canary.CampaignCallbackContext(cfg,{})
    cfg["original_wh"]=[1,1]
    with pytest.raises(RuntimeError,match="campaign_callback_context_mismatch"):
        canary.NativeLibrary(sdk_library).run_campaign(cfg,out_json=tmp_path / "runtime.json",
            context=context,runtime_instance_id="failed-fixture")
    assert context.measured.completed_count == 0
    report=json.loads((tmp_path / "runtime.json").read_text())
    assert report["ok"] is False and report["callback_failures"] == 4


def energy_binding(cfg, library, canary):
    identity={"model_id":"yolov7_paper","case_id":"b066","setup_id":"software_h8",
              "precision":"uint8_dequant_fp16","eval_run_id":"software","source_run_id":"software"}
    cfg.update(identity=identity,quality_binding_sha256="1"*64,
               energy_workload_only=True,duration_s=60.0,warmup=0,repetitions=1,frames=1)
    specs={key:copy.deepcopy(cfg[key]) for key in ("hef","engine","prepared_input")}
    specs["native_executable"]={"path":str(library),"sha256":canary.sha256_file(library),"size_bytes":library.stat().st_size}
    records=[]
    for spec in specs.values():
        st=Path(spec["path"]).stat()
        records.append({"path":spec["path"],"size_bytes":st.st_size,"mtime_ns":st.st_mtime_ns,"device":st.st_dev,"inode":st.st_ino})
    boundary=cfg["boundary_contract"]
    return {"backend":"hailo8_to_trt","workload_supported":True,"model":"yolov7_paper","case":"b066",
        "setup_id":"software_h8","precision":"uint8_dequant_fp16","eval_run_id":"software","source_run_id":"software",
        "native_split_quality_binding_sha256":"1"*64,"input_image":cfg["input_image"]["path"],
        "input_image_sha256":cfg["input_image"]["sha256"],"artifacts":specs,"verified_files":records,
        "prepared_input_contract":copy.deepcopy(cfg["prepared_input"]),
        "runtime_options":{"completion_execution_contract":cfg["completion_contract"],"queue_depth":3,"post_queue_depth":4,"device_id":""},
        "runtime_boundary_evidence":{"output_name":boundary["name"],"output_shape":boundary["shape"],"output_dtype":"uint8",
            "trt_input_name":boundary["trt_input"]["name"],"trt_input_shape":boundary["trt_input"]["shape"],"trt_input_dtype":"uint8"}}


@pytest.mark.parametrize("damage",["changed_file","missing_stat","library_stat","engine_alias","completion","identity","input","budget"])
def test_energy_requires_unchanged_normal_preflight_binding(canary,sdk_library,monkeypatch,tmp_path,damage):
    cfg=campaign_config(tmp_path / "inputs",canary)
    binding=energy_binding(cfg,sdk_library,canary)
    if damage=="changed_file": Path(cfg["engine"]["path"]).write_bytes(b"changed")
    elif damage=="missing_stat": binding["verified_files"]=[]
    elif damage=="library_stat": binding["verified_files"].pop()
    elif damage=="engine_alias": cfg["engine"]["path"]="foreign.engine"
    elif damage=="completion": binding["runtime_options"]["completion_execution_contract"]={}
    elif damage=="identity": cfg["identity"]["model_id"]="foreign"
    elif damage=="input": cfg["prepared_input"]["name"]="foreign"
    elif damage=="budget": cfg["warmup"]=100
    monkeypatch.setattr(canary,"NativeLibrary",lambda _:pytest.fail("runtime reached"))
    monkeypatch.setattr(canary,"build_cpp",lambda _:pytest.fail("build reached"))
    monkeypatch.setattr(canary,"sha256_file",lambda _:pytest.fail("heavy artifact hash reached"))
    with pytest.raises(RuntimeError):
        canary.run_energy_campaign(cfg,out_root=tmp_path / "energy",verified_workload_binding=binding)


def test_normal_energy_runtime_sixty_seconds_without_build_hash_or_tensor_dump(canary,sdk_library,monkeypatch,tmp_path):
    cfg=campaign_config(tmp_path / "inputs",canary)
    binding=energy_binding(cfg,sdk_library,canary)
    monkeypatch.setattr(canary,"build_cpp",lambda _:pytest.fail("energy runtime build"))
    monkeypatch.setattr(canary,"sha256_file",lambda _:pytest.fail("energy heavy artifact hash"))
    monkeypatch.setattr(canary.np,"save",lambda *args,**kw:pytest.fail("energy tensor dump"))
    report=canary.run_energy_campaign(cfg,out_root=tmp_path / "energy",verified_workload_binding=binding)
    assert report["ok"] is True and len(report["repetitions"]) == 1
    row=report["repetitions"][0]
    assert row["makespan_ms"] >= 60000
    assert row["completed_work_units"] > cfg["frames"]
    assert row["warmup_completed_frames"] == 0 and row["output_arrays"] == {} and row["boundary_array"] is None
    assert row["completion_attestation"]["status"] == "passed"
    assert row["completion_attestation"]["completed_work_units"] == row["completed_work_units"]
    assert not list((tmp_path / "energy").glob("last_*"))
