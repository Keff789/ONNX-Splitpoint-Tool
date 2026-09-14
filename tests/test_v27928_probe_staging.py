"""Offline regressions; synthetic model/template and simulated SSH, NO hardware."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


collector = load("probe_fix_collector", ROOT / "scripts/deepx_full_output_probe_v27928.py")
worker = load("probe_fix_worker", ROOT / "scripts/deepx_full_output_probe_worker_v27928.py")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, dict) or isinstance(data, list):
        path.write_text(json.dumps(data), encoding="utf-8")
    elif isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_text(data, encoding="utf-8")


TEMPLATE = '''from pathlib import Path
import json

def run_deepx_output_value_probe(root, dxnn, run, args, output):
    # Synthetic output only: this test does not contain or call DXRT.
    import splitpoint_runners
    assert splitpoint_runners.FIXTURE is True
    assert Path(splitpoint_runners.__file__).is_relative_to(root)
    assert dxnn == root / "deepx/deepx_m1/full/model.dxnn"
    assert args.runs == 1 and args.warmup == 0
    assert args.validation_images == "" and args.validation_max_images == 0
    assert Path(args.prepared_feed_image).is_relative_to(root)
    assert run["model_id"] == "yolo11l"
    assert run["setup_id"] == "orin_nx_deepx_m1_01"
    assert run["quality_endpoint"] == "completed_detection"
    original = json.loads((root / "deepx/deepx_m1/full/output_contract.json").read_text())
    assert original["input"]["dtype"] == "uint8"
    assert original["input"]["letterbox_pad_value"] == 114
    (output / "single_call.txt").write_text("1")
    return {"status": "invalid_output_values_captured", "diagnostic_only": True,
            "synthetic_fixture": True, "error": "decoded_pre_nms_values_invalid"}
'''

# No external commands are contacted. All files are synthetic and temporary.
SSH = '''#!/usr/bin/env python3
import json,os,pathlib,shlex,shutil,subprocess,sys,tempfile,secrets,string
command = sys.argv[-1]
log = pathlib.Path(os.environ["FAKE_TRANSPORT_LOG"])
with log.open("a") as f: f.write(json.dumps({"tool":"ssh","command":command})+"\\n")
words = shlex.split(command)
if words[0] == "mktemp":
    path = "/tmp/onnx-v27928-full-probe-" + "".join(secrets.choice(string.ascii_letters+string.digits) for _ in range(10))
    os.mkdir(path,0o700); print(path)
elif words[0] == "rm":
    assert words[:3] == ["rm","-rf","--"]
    assert words[3].startswith("/tmp/onnx-v27928-full-probe-")
    shutil.rmtree(words[3])
elif words[0] == "timeout":
    if os.environ.get("FAKE_TIMEOUT") == "1":
        work = pathlib.Path(words[words.index("--request")+1]).parent
        (work/"results").mkdir()
        sys.exit(124)
    sys.exit(subprocess.run(words).returncode)
else:
    raise RuntimeError("unexpected SSH operation")
'''
SCP = '''#!/usr/bin/env python3
import json,os,pathlib,shutil,sys
args=sys.argv[1:]; paths=[]; i=0
while i<len(args):
    if args[i] in ("-o","-P"): i+=2
    elif args[i] in ("-q","-r"): i+=1
    else: paths.append(args[i]); i+=1
with pathlib.Path(os.environ["FAKE_TRANSPORT_LOG"]).open("a") as f:
    f.write(json.dumps({"tool":"scp","paths":paths})+"\\n")
def local(s):
    return pathlib.Path(s.split(":",1)[1] if "@" in s and ":" in s else s)
destination=local(paths[-1])
for value in paths[:-1]:
    source=local(value)
    target=destination/source.name if destination.is_dir() else destination
    if source.is_dir(): shutil.copytree(source,target)
    else: shutil.copyfile(source,target)
'''


class ProbeFixTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="probe-fix-test-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.run = self.base / collector.DEFAULT_RUN
        self.suite = self.run / "models/yolo11l/benchmark_set/legacy_suite"
        self.bundle = self.base / collector.DEFAULT_BUNDLE
        self.output = self.base / "diagnostics"
        self.image_name = "000000212226.jpg"
        self.image_rel = "resources/validation/detection/val2017_n500_s20260710/" + self.image_name
        self.old = "/home/nonexistent-probe-fixture/deleted-remote-run/1/suite"
        self.model_bytes = b"SYNTHETIC_MODEL_NOT_A_DXNN"
        self.image_bytes = b"SYNTHETIC_IMAGE_NOT_A_JPEG"
        self.cache = self.base / "BackendArtifacts/deepx/synthetic/model.dxnn"
        self.canonical_image = self.base / "dataset/val2017" / self.image_name
        self.contract = {"model_id":"yolo11l", "endpoint_mode":"decoded_pre_nms",
                         "artifact_path":str(self.cache), "artifact_sha256":digest(self.model_bytes),
                         "suite_artifact_sha256":digest(self.model_bytes),
                         "input":{"dtype":"uint8","letterbox_pad_value":114}}
        self.plan = {"runs":[{"id":"deepx_m1_full","setup_id":"orin_nx_deepx_m1_01",
                              "dxnn_path":collector.DXNN_RELATIVE,"contract_path":collector.CONTRACT_RELATIVE,
                              "quality_endpoint":"completed_detection"}]}
        self.venv = self.base / "runtime_venv"
        (self.venv / "bin").mkdir(parents=True)
        # A symlink to a copied venv interpreter cannot discover its standard
        # library alone when the real worker restarts with -I. Bind this fake
        # runtime like an ordinary venv; keep the isolated worker unchanged.
        (self.venv / "pyvenv.cfg").write_text(
            "home = " + str(Path(sys._base_executable).resolve().parent)
            + "\ninclude-system-site-packages = true\n",
            encoding="utf-8",
        )
        (self.venv / "bin/python").symlink_to(Path(sys._base_executable).resolve())
        put(self.run / "hardware_matrix.json", {"hardware_targets":[{
            "id":"orin_nx_deepx_m1_01","accelerator":"deepx_m1",
            "runtime":{"host":"192.168.0.102","user":"nx","port":22},
            "build_environment":{"runtime_venv":str(self.venv)}}]})
        put(self.run / "models/yolo11l/benchmark_results/benchmark_results_deepx_m1_full_auto.json", [{
            "run_id":"deepx_m1_full","backend":"deepx_m1","variant":"full",
            "dxnn_path":self.old + "/" + collector.DXNN_RELATIVE,
            "deepx_prepared_feed_benchmark":{"image":self.old + "/" + self.image_rel}}])
        put(self.suite / collector.CONTRACT_RELATIVE, self.contract)
        put(self.suite / "benchmark_plan.json", self.plan)
        put(self.suite / "output_contracts.json", {"do_not_change":True})
        put(self.suite / collector.DXNN_RELATIVE, self.model_bytes)
        put(self.cache, self.model_bytes)
        put(self.suite / self.image_rel, self.image_bytes)
        put(self.canonical_image, self.image_bytes)
        put(self.run / "campaign/inputs/dataset_detection_validation.json", {
            "root":str(self.canonical_image.parent),"items":[{"relative_path":self.image_name,
                                                              "sha256":"sha256:"+digest(self.image_bytes)}]})
        for name in ("__init__.py","harness/base.py","harness/yolo.py"):
            put(self.suite / "splitpoint_runners" / name, "FIXTURE = True\n")
        put(self.bundle / "probe/benchmark_suite.py.txt", TEMPLATE)
        self.args = argparse.Namespace(run_dir=self.run, model="yolo11l", bundle_dir=self.bundle,
                                       payload_dir=None, local_dxnn=None, local_image=None)
        self.bin = self.base / "bin"
        self.bin.mkdir()
        for name, script in (("ssh", SSH),("scp", SCP)):
            path = self.bin / name
            path.write_text(script)
            path.chmod(0o755)
        self.log = self.base / "fake_transport.jsonl"
        self.addCleanup(self.cleanup_fake_remotes)

    def cleanup_fake_remotes(self):
        if self.log.is_file():
            for line in self.log.read_text().splitlines():
                row=json.loads(line)
                for token in row.get("paths",[]):
                    if ":/tmp/onnx-v27928-full-probe-" in token:
                        path=Path(token.split(":",1)[1].rstrip("/"))
                        if path.name == "results": path=path.parent
                        if collector.remote_temp_valid(str(path)) and path.exists(): shutil.rmtree(path)

    def resolved(self):
        return collector.resolve_inputs(self.args)

    def stage(self):
        request, remote, sources = self.resolved()
        stage = self.base / "fresh_remote"
        stage.mkdir()
        collector.prepare_stage(stage, request, collector.staging_files(request,sources))
        return stage, request

    def execute(self, *options, env_add=None):
        env=dict(os.environ)
        env["PATH"]=str(self.bin)+os.pathsep+env["PATH"]
        env["FAKE_TRANSPORT_LOG"]=str(self.log)
        env.update(env_add or {})
        return subprocess.run([sys.executable,"-I","-B",str(ROOT/"scripts/deepx_full_output_probe_v27928.py"),
                               "--run-dir",str(self.run),"--bundle-dir",str(self.bundle),
                               "--output-dir",str(self.output),*options],
                              text=True,capture_output=True,timeout=30,env=env)

    def summary(self):
        files=list(self.output.glob("*.zip"))
        self.assertEqual(len(files),1)
        with zipfile.ZipFile(files[0]) as z:
            self.assertIsNone(z.testzip())
            return json.loads(z.read("collection_summary.json")), set(z.namelist())

    def test_01_resolve_does_not_touch_old_remote_directory(self):
        request, remote, sources=self.resolved()
        self.assertEqual(request["original_remote_suite"],self.old)
        self.assertEqual(request["staging_mode"],collector.STAGING_MODE)
        self.assertNotIn("remote_suite",request)
        self.assertEqual(remote["host"],"192.168.0.102")

    def test_02_full_dxnn_from_recorded_cache_when_local_suite_copy_missing(self):
        (self.suite/collector.DXNN_RELATIVE).unlink()
        self.assertEqual(self.resolved()[2]["dxnn"],self.cache)

    def test_03_wrong_model_rejected(self):
        put(self.suite/collector.DXNN_RELATIVE,b"wrong model")
        self.cache.unlink()
        with self.assertRaisesRegex(ValueError,"sha256 mismatch"): self.resolved()

    def test_04_exact_original_image_from_canonical_manifest(self):
        (self.suite/self.image_rel).unlink()
        self.assertEqual(self.resolved()[2]["image"],self.canonical_image)

    def test_05_wrong_image_rejected_without_silent_replacement(self):
        put(self.suite/self.image_rel,b"wrong image")
        self.canonical_image.unlink()
        with self.assertRaisesRegex(ValueError,"sha256 mismatch"): self.resolved()

    def test_06_other_filename_cannot_be_forced(self):
        self.args.local_image=self.base/"another.jpg"
        with self.assertRaisesRegex(ValueError,"must_be_the_original"): self.resolved()

    def test_07_missing_local_runner_fails_before_ssh(self):
        (self.suite/"splitpoint_runners/harness/yolo.py").unlink()
        result=self.execute()
        self.assertEqual(result.returncode,2)
        self.assertFalse(self.log.exists())
        summary,names=self.summary()
        self.assertIn("local_generated_runner_missing",summary["error"])
        self.assertIn("collector_traceback.log",names)

    def test_08_wrong_template_fails_before_ssh(self):
        put(self.bundle/"probe/benchmark_suite.py.txt","# no probe API\n")
        result=self.execute()
        self.assertEqual(result.returncode,2)
        self.assertFalse(self.log.exists())
        self.assertIn("probe_api_missing",self.summary()[0]["error"])

    def test_09_plan_only_is_readonly_and_no_ssh(self):
        result=self.execute("--plan-only")
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertFalse(self.log.exists())
        self.assertFalse(self.output.exists())
        self.assertIs(json.loads(result.stdout)["ssh_executed"],False)

    def test_10_stage_contains_dependencies_without_whole_dataset_or_weights(self):
        put(self.suite/"models/yolo11l.onnx",b"EXCLUDED_ONNX")
        put(self.suite/"b003/model.dxnn",b"EXCLUDED_PART1")
        put(self.suite/str(Path(self.image_rel).parent)/"another.jpg",b"EXCLUDED_IMAGE")
        stage,request=self.stage()
        self.assertFalse((stage/"suite/models/yolo11l.onnx").exists())
        self.assertFalse((stage/"suite/b003").exists())
        self.assertEqual(len(list(stage.rglob("*.jpg"))),1)
        self.assertEqual((stage/"suite"/collector.CONTRACT_RELATIVE).read_bytes(),(self.suite/collector.CONTRACT_RELATIVE).read_bytes())
        self.assertEqual((stage/"suite/splitpoint_runners/native_detection_postprocess.py").read_bytes(),(ROOT/"onnx_splitpoint_tool/native_detection_postprocess.py").read_bytes())
        worker.validate_stage(request,stage)

    def test_11_no_hardlink_to_original(self):
        stage,request=self.stage()
        staged=stage/"suite"/collector.DXNN_RELATIVE
        self.assertNotEqual(staged.stat().st_ino,(self.suite/collector.DXNN_RELATIVE).stat().st_ino)
        staged.write_bytes(b"changed staged model")
        self.assertEqual((self.suite/collector.DXNN_RELATIVE).read_bytes(),self.model_bytes)

    def test_12_worker_rejects_transferred_corruption(self):
        stage,request=self.stage()
        (stage/"suite"/collector.DXNN_RELATIVE).write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError,"staged_dxnn_sha256_mismatch"):
            worker.validate_stage(request,stage)

    def test_13_worker_rejects_old_remote_request(self):
        with self.assertRaisesRegex(ValueError,"staged_probe_request_required"):
            worker.validate_stage({"schema":collector.REQUEST_SCHEMA,"remote_suite":self.old},self.base)

    def test_14_worker_rejects_path_escape(self):
        stage,request=self.stage()
        request["staged_dxnn_relative"]="../../outside.dxnn"
        with self.assertRaisesRegex(ValueError,"relative_path_invalid"):
            worker.validate_stage(request,stage)

    def test_15_setup_returncode_cannot_be_masked_by_collection(self):
        self.assertEqual(collector.exit_code(collected=True,remote_returncode=2,result={"status":"probe_setup_failed"}),2)
        self.assertEqual(collector.exit_code(collected=True,remote_returncode=0,result={"status":"probe_setup_failed"}),2)
        self.assertEqual(collector.exit_code(collected=False,remote_returncode=0,result={"status":"captured"}),2)

    def test_16_full_simulated_transport_worker_zip_cleanup(self):
        before={str(p):digest(p.read_bytes()) for p in self.run.rglob("*") if p.is_file()}
        result=self.execute()
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        summary,names=self.summary()
        self.assertEqual(summary["probe_status"],"invalid_output_values_captured")
        self.assertEqual(summary["remote_returncode"],0)
        self.assertIs(summary["remote_temporary_directory_removed"],True)
        self.assertFalse(Path(summary["remote_probe_dir"]).exists())
        self.assertIn("results/single_call.txt",names)
        self.assertFalse(any(name.endswith(".dxnn") for name in names))
        self.assertFalse(list(self.output.glob(".deepx-probe-transfer-*")))
        after={str(p):digest(p.read_bytes()) for p in self.run.rglob("*") if p.is_file()}
        self.assertEqual(before,after)
        for row in map(json.loads,self.log.read_text().splitlines()):
            self.assertNotIn(self.old,str(row))

    def test_17_missing_runtime_reports_zip_and_nonzero_exit(self):
        (self.venv/"bin/python").unlink()
        result=self.execute()
        self.assertEqual(result.returncode,2,result.stdout+result.stderr)
        summary,names=self.summary()
        self.assertEqual(summary["probe_status"],"probe_setup_failed")
        self.assertEqual(summary["collector_exit_code"],2)
        self.assertIn("existing_deepx_runtime_python_missing",summary["probe_error"])
        self.assertIn("results/probe_traceback.log",names)

    def test_18_timeout_does_not_pass(self):
        result=self.execute(env_add={"FAKE_TIMEOUT":"1"})
        self.assertEqual(result.returncode,2,result.stdout+result.stderr)
        summary,names=self.summary()
        self.assertEqual(summary["probe_status"],"probe_timeout")
        self.assertEqual(summary["remote_returncode"],124)
        self.assertEqual(summary["collector_exit_code"],2)

    def test_19_cleanup_path_validation(self):
        self.assertTrue(collector.remote_temp_valid("/tmp/onnx-v27928-full-probe-AbCd123456"))
        for path in ("/tmp","/","/home/kmika/Models","/tmp/onnx-v27928-full-probe-AbCd123456/..",self.old):
            self.assertFalse(collector.remote_temp_valid(path))

    def test_20_changed_setup_rejected_remotely(self):
        stage,request=self.stage()
        request["setup_id"]="other_setup"
        with self.assertRaisesRegex(ValueError,"setup_mismatch"):
            worker.validate_stage(request,stage)

    def test_21_source_file_change_during_staging_is_detected(self):
        request,remote,sources=self.resolved()
        files=collector.staging_files(request,sources)
        sources["image"].write_bytes(b"image replaced after preflight")
        stage=self.base/"stage";stage.mkdir()
        with self.assertRaisesRegex(ValueError,"staged_image_sha256_mismatch"):
            collector.prepare_stage(stage,request,files)

    def test_22_runner_symlink_cannot_pull_external_files_into_payload(self):
        external=self.base/"external.py";external.write_text("SECRET=1")
        (self.suite/"splitpoint_runners/external.py").symlink_to(external)
        request,remote,sources=self.resolved()
        with self.assertRaisesRegex(ValueError,"outside_generated_package"):
            collector.staging_files(request,sources)

    def test_23_suite_root_python_companions_are_staged(self):
        put(self.suite/"scientific_reporter_v60.py","# optional original companion\n")
        stage,request=self.stage()
        self.assertEqual((stage/"suite/scientific_reporter_v60.py").read_bytes(),
                         (self.suite/"scientific_reporter_v60.py").read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
