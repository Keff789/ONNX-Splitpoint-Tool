"""Central validation digest sharing; Native source/default behavior stays intact."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
from threading import Event

import pytest

from onnx_splitpoint_tool import generic_only_plan as admission
from onnx_splitpoint_tool.runners import generic_quality_completion as verifier


def test_real_central_completion_keeps_native_bytes_and_hashes_once_then_rechecks(tmp_path, monkeypatch):
    from test_generic_quality_completion import make_config, envelope
    fixture = make_config(tmp_path)
    request, result, request_path, _ = envelope(fixture, tmp_path)
    native = verifier._product()
    native_path = Path(native.__file__)
    before = native_path.read_bytes()
    paths = {native_path.resolve(), Path(native._yolo_module.__file__).resolve(),
             Path(native._base_module.__file__).resolve()}
    calls = Counter()
    original = admission._file_sha256
    def counted(path):
        calls[Path(path).resolve()] += 1
        return original(path)
    monkeypatch.setattr(admission, "_file_sha256", counted)
    # Ensure test isolation even though the real product installs only once.
    monkeypatch.setattr(native, "file_sha256", native.file_sha256)
    args = (request, result, fixture["config"]["completion_contract"],
            fixture["config"]["expected_runtime_artifacts"])
    kwargs = {"request_path": request_path, "reference_descriptor": {}}
    with admission.validation_scope():
        first = admission._completed_quality_binding(*args, **kwargs)
        second = admission._completed_quality_binding(*args, **kwargs)
        assert first == second and first["all_actual_completed_records_verified"] is True
        assert all(calls[path] == 1 for path in paths)
    assert all(calls[path] == 2 for path in paths)
    assert native_path.read_bytes() == before


@pytest.mark.parametrize("late", [False, True])
def test_same_size_same_mtime_change_never_reuses_digest(tmp_path, monkeypatch, late):
    native = verifier._product()
    monkeypatch.setattr(native, "file_sha256", native.file_sha256)
    source = tmp_path / "bound-source.py"
    source.write_bytes(b"original source")
    before = source.stat()
    monkeypatch.setattr(verifier, "validate_completed_quality_binding",
                        lambda: native.file_sha256(source))
    with pytest.raises(ValueError, match="source changed"):
        with admission.validation_scope():
            admission._completed_quality_binding()
            source.write_bytes(b"modified source")
            os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
            if not late:
                admission._completed_quality_binding()


def test_final_full_rehash_catches_change_even_if_metadata_appears_unchanged(tmp_path, monkeypatch):
    native = verifier._product()
    monkeypatch.setattr(native, "file_sha256", native.file_sha256)
    source = tmp_path / "bound-source.py"
    source.write_bytes(b"original source")
    version = admission._file_version(source)
    monkeypatch.setattr(admission, "_file_version", lambda path: version)
    monkeypatch.setattr(verifier, "validate_completed_quality_binding",
                        lambda: native.file_sha256(source))
    with pytest.raises(ValueError, match="source changed"):
        with admission.validation_scope():
            admission._completed_quality_binding()
            source.write_bytes(b"modified source")


def test_other_thread_and_calls_outside_central_context_keep_original_reader(tmp_path, monkeypatch):
    native = verifier._product()
    original = native.file_sha256
    native_reads = Counter()
    def counted(path):
        native_reads["reads"] += 1
        return original(path)
    monkeypatch.setattr(native, "file_sha256", counted)
    source = tmp_path / "bound-source.py"
    source.write_bytes(b"source")
    expected = hashlib.sha256(b"source").hexdigest()
    entered, complete = Event(), Event()
    def default_reader():
        assert entered.wait(5)
        assert native.file_sha256(source) == expected
        assert native.file_sha256(source) == expected
        complete.set()
    def central():
        assert native.file_sha256(source) == expected
        entered.set()
        assert complete.wait(5)
        return native.file_sha256(source)
    monkeypatch.setattr(verifier, "validate_completed_quality_binding", central)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(default_reader)
        with admission.validation_scope():
            assert admission._completed_quality_binding() == expected
            assert native_reads["reads"] == 2
            # A GQP scope alone must not activate the adapter.
            assert native.file_sha256(source) == expected
            assert native_reads["reads"] == 3
        future.result(timeout=5)
    assert native.file_sha256(source) == expected
    assert native_reads["reads"] == 4


def test_parallel_groups_have_independent_version_scopes(tmp_path, monkeypatch):
    native = verifier._product()
    monkeypatch.setattr(native, "file_sha256", native.file_sha256)
    source = tmp_path / "bound-source.py"
    source.write_bytes(b"source")
    calls = Counter()
    original = admission._file_sha256
    def counted(path):
        from threading import get_ident
        calls[get_ident()] += 1
        return original(path)
    monkeypatch.setattr(admission, "_file_sha256", counted)
    monkeypatch.setattr(verifier, "validate_completed_quality_binding",
                        lambda: [native.file_sha256(source) for _ in range(5)])
    one, two = Event(), Event()
    def group(ready, other):
        with admission.validation_scope():
            values = admission._completed_quality_binding()
            ready.set()
            assert other.wait(5)
        return values
    with ThreadPoolExecutor(max_workers=2) as executor:
        a = executor.submit(group, one, two)
        b = executor.submit(group, two, one)
        assert a.result(timeout=5) == b.result(timeout=5)
    assert len(calls) == 2 and set(calls.values()) == {2}


def test_installation_is_serialized_idempotent_and_exception_resets_context(tmp_path, monkeypatch):
    native = verifier._product()
    original = native.file_sha256
    monkeypatch.setattr(native, "file_sha256", original)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda _: admission._install_native_digest_adapter(native), range(12)))
    installed = native.file_sha256
    assert installed.__wrapped__ is original
    admission._install_native_digest_adapter(native)
    assert native.file_sha256 is installed
    def failure():
        raise ValueError("actual verifier failure")
    monkeypatch.setattr(verifier, "validate_completed_quality_binding", failure)
    with admission.validation_scope():
        with pytest.raises(ValueError, match="actual verifier failure"):
            admission._completed_quality_binding()
        assert admission._NATIVE_DIGEST_SCOPE.get() is False


def test_actual_bad_completion_contract_still_fails(tmp_path, monkeypatch):
    from test_generic_quality_completion import make_config, envelope
    fixture = make_config(tmp_path)
    request, result, request_path, _ = envelope(fixture, tmp_path)
    native = verifier._product()
    monkeypatch.setattr(native, "file_sha256", native.file_sha256)
    result["producer_identity_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="normal result/reference/image binding"):
        with admission.validation_scope():
            admission._completed_quality_binding(request, result,
                fixture["config"]["completion_contract"], fixture["config"]["expected_runtime_artifacts"],
                request_path=request_path, reference_descriptor={})
