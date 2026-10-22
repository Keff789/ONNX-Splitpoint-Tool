"""CPU-only ownership checks for the real untimed input observer.

No SDK/session constructor, inference engine, contract bypass or production
garbage-collector policy is used. This does not reproduce a native SIGSEGV.
"""
from __future__ import annotations

import gc
import types
import weakref

import numpy as np
import pytest

from onnx_splitpoint_tool.runners.generic_quality_completion import GenericQualityCompletion


def observer():
    # Exercise only the actual observation method, not artifact/SDK creation.
    return GenericQualityCompletion.__new__(GenericQualityCompletion)


class Session:
    def __init__(self):
        self.calls = []
        self.result = object()
        self.error = None

    def infer(self, inputs, *args, **kwargs):
        self.calls.append((inputs, args, kwargs))
        if self.error is not None:
            raise self.error
        return self.result

    def close(self):
        raise AssertionError("The observer must not perform SDK teardown")


def feed():
    return {"images": np.arange(6, dtype=np.float32).reshape(2, 3)}


def test_class_descriptor_restored_and_arguments_results_snapshots_preserved():
    owner, session, inputs = observer(), Session(), feed()
    descriptor = Session.__dict__["infer"]
    assert "infer" not in session.__dict__
    with owner.observe_stage1_inputs(session, prepared_inputs=inputs):
        assert session.infer(inputs, "positional", token="value") is session.result
        inputs["images"][0, 0] = 99
    assert "infer" not in session.__dict__
    assert Session.__dict__["infer"] is descriptor
    assert session.infer.__func__ is descriptor
    assert len(session.calls) == 1
    actual_inputs, args, kwargs = session.calls[0]
    assert actual_inputs is inputs and args == ("positional",) and kwargs == {"token": "value"}
    assert owner._runtime_inputs["images"][0, 0] == 0
    assert owner._prepared_input_snapshot["images"] == owner._runtime_input_snapshot["images"]


@pytest.mark.parametrize("bound_override", [False, True])
def test_exact_preexisting_instance_override_restored(bound_override):
    owner, session, inputs = observer(), Session(), feed()
    calls, result = [], object()

    def custom(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    override = types.MethodType(custom, session) if bound_override else custom
    session.infer = override
    with owner.observe_stage1_inputs(session):
        assert session.infer(inputs, marker=7) is result
    assert session.__dict__["infer"] is override
    assert session.infer is override
    assert len(calls) == 1 and not session.calls
    expected = (session, inputs) if bound_override else (inputs,)
    assert calls[0] == (expected, {"marker": 7})


@pytest.mark.parametrize("failure", ["inference", "caller_before", "caller_after", "no_call", "double_call"])
@pytest.mark.parametrize("override", [False, True])
def test_original_attribute_state_restored_on_every_exit(failure, override):
    owner, session, inputs = observer(), Session(), feed()
    if override:
        session.infer = session.infer
    original_instance = dict(session.__dict__)
    inference_error, caller_error = RuntimeError("SDK leaf"), RuntimeError("caller leaf")
    if failure == "inference":
        session.error = inference_error

    def execute():
        with owner.observe_stage1_inputs(session):
            if failure == "caller_before":
                raise caller_error
            if failure == "no_call":
                return
            session.infer(inputs)
            if failure == "caller_after":
                raise caller_error
            if failure == "double_call":
                session.infer(inputs)

    expected = ValueError if failure in {"no_call", "double_call"} else RuntimeError
    with pytest.raises(expected) as caught:
        execute()
    if failure == "inference":
        assert caught.value is inference_error
    elif failure.startswith("caller"):
        assert caught.value is caller_error
    elif failure == "no_call":
        assert "inference was not observed" in str(caught.value)
    else:
        assert "exactly one actual stage1 inference" in str(caught.value)
    assert ("infer" in session.__dict__) is override
    if override:
        assert session.__dict__["infer"] is original_instance["infer"]
    assert len(session.calls) == (0 if failure in {"no_call", "caller_before"} else 1)


def test_repeated_observation_does_not_accumulate_instance_method_shadow():
    owner, session, inputs = observer(), Session(), feed()
    for _ in range(3):
        with owner.observe_stage1_inputs(session):
            assert session.infer(inputs) is session.result
        assert "infer" not in session.__dict__
    assert len(session.calls) == 3


@pytest.mark.parametrize("failure", [None, "inference", "caller", "no_call", "double_call"])
def test_no_new_session_cycle_requires_no_cyclic_gc_to_release(failure):
    owner = observer()

    def run_one():
        session, inputs = Session(), feed()
        reference = weakref.ref(session)
        if failure == "inference":
            session.error = RuntimeError("SDK leaf")
        try:
            with owner.observe_stage1_inputs(session):
                if failure == "caller":
                    raise RuntimeError("caller leaf")
                if failure != "no_call":
                    session.infer(inputs)
                if failure == "double_call":
                    session.infer(inputs)
        except (RuntimeError, ValueError):
            assert failure is not None
            # Do not retain the fixture's own exception->traceback->self cycle.
            session.error = None
        return reference

    enabled = gc.isenabled()
    gc.collect()
    gc.disable()  # Deterministic test only; the product never changes GC.
    try:
        reference = run_one()
        assert reference() is None, "Observer must not defer native-wrapper release to cyclic GC"
    finally:
        if enabled:
            gc.enable()


def test_slot_only_existing_override_keeps_previous_compatibility():
    class SlotSession:
        __slots__ = ("infer",)

    session, owner, inputs = SlotSession(), observer(), feed()
    original = lambda value: value
    session.infer = original
    with owner.observe_stage1_inputs(session):
        assert session.infer(inputs) is inputs
    assert session.infer is original
