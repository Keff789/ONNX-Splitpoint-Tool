"""Bound original-image feeds for Generic P2, prepared outside every timer."""
from collections.abc import Mapping
from copy import deepcopy
import hashlib
from pathlib import Path
import re

import numpy as np

SCHEMA = "onnx-splitpoint/generic-canonical-full-input/v1"


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot(array):
    value = np.asarray(array)
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "tensor_sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}


def _content(value):
    if not isinstance(value, Mapping):
        raise ValueError("Canonical original-input snapshot missing")
    result = {key: deepcopy(value.get(key)) for key in ("shape", "dtype", "tensor_sha256")}
    if (not isinstance(result["shape"], list) or not result["shape"]
            or any(type(dim) is not int or dim <= 0 for dim in result["shape"])
            or not isinstance(result["dtype"], str)
            or re.fullmatch(r"[a-f0-9]{64}", str(result["tensor_sha256"])) is None):
        raise ValueError("Canonical original-input snapshot malformed")
    return result


def passthrough_sources(config):
    boundary = config.get("boundary_observation") or {}
    producer_boundary = (config.get("generic_artifact_binding") or {}).get("boundary_contract") or {}
    if producer_boundary.get("passthrough_inputs") and producer_boundary != boundary:
        raise ValueError("Actual original-input passthrough cannot drop or replace its bound boundary observation")
    mapping = boundary.get("passthrough_inputs") or {}
    if not isinstance(mapping, Mapping):
        raise ValueError("Generic original-input passthrough mapping malformed")
    return set(mapping.values())


def binding_from_quality(proof, image_id, source_image, prepared_input_sha256, producer_source):
    matches = [entry for entry in proof["per_image"] if entry.get("image_id") == image_id]
    if len(matches) != 1:
        raise ValueError("Canonical original input lacks one exact actual Quality image")
    image = matches[0]
    return {"schema": SCHEMA, "image_id": image_id, "source_image": deepcopy(source_image),
            "prepared_input_sha256": prepared_input_sha256, "producer_source": deepcopy(producer_source),
            "preprocessing_contract": deepcopy(proof["image_preprocessing_contract"]),
            "inputs": {name: _content(value) for name, value in image["prepared_inputs"].items()},
            "runtime_inputs": {name: _content(value) for name, value in image["runtime_inputs"].items()}}


def validate_quality_bindings(row, proof):
    """Rejoin every canonical byte expectation to the actual normal Quality."""
    sources = passthrough_sources(row)
    bindings = row.get("canonical_full_input_bindings")
    if not sources:
        if bindings is not None:
            raise ValueError("Canonical full-input bindings require an actual passthrough")
        return
    if not isinstance(bindings, list) or len(bindings) != 2:
        raise ValueError("Actual image passthrough requires two bound canonical full inputs")
    observed = (proof.get("runtime_context") or {}).get("boundary_runtime") or {}
    if observed.get("passthrough_inputs") != row["boundary_observation"]["passthrough_inputs"]:
        raise ValueError("Canonical original-input mapping differs from actual normal Quality")
    for index, binding in enumerate(bindings):
        if (not isinstance(binding, Mapping) or binding.get("schema") != SCHEMA
                or binding.get("producer_source") != row["quality_reference"]["request"]
                or binding.get("preprocessing_contract") != proof["image_preprocessing_contract"]
                or set(binding.get("inputs") or {}) != sources
                or re.fullmatch(r"[a-f0-9]{64}", str(binding.get("prepared_input_sha256"))) is None):
            raise ValueError("Canonical original-input binding differs from its actual Quality source")
        images = [entry for entry in proof["per_image"] if entry.get("image_id") == binding.get("image_id")]
        if len(images) != 1:
            raise ValueError("Canonical original-input Quality image is missing or ambiguous")
        image = images[0]
        descriptor = binding.get("source_image") or {}
        local = Path(str(descriptor.get("local_path") or descriptor.get("path") or ""))
        if (not local.is_absolute() or not local.is_file()
                or descriptor.get("sha256") != image["source_image_sha256"]
                or _sha(local) != descriptor.get("sha256")
                or binding.get("runtime_inputs") != {name: _content(value) for name, value in image["runtime_inputs"].items()}
                or any(_content(binding["inputs"][name]) != _content(image["prepared_inputs"].get(name)) for name in sources)):
            raise ValueError("Canonical original-input image or actual Quality tensor bytes differ")
        for target, name in row["boundary_observation"]["passthrough_inputs"].items():
            spec = row["boundary_observation"]["part2_inputs"][target]
            if any(binding["inputs"][name][key] != spec[key] for key in ("shape", "dtype")):
                raise ValueError("Canonical original input differs from the actual P2 passthrough contract")
        if index == 0 and (binding["image_id"] != proof["representative_input"]["image_id"]
                or binding["prepared_input_sha256"] != row["input_binding"]["prepared_input"]["sha256"]):
            raise ValueError("Canonical baseline differs from the unchanged timed input")
    if bindings[0]["image_id"] == bindings[1]["image_id"]:
        raise ValueError("Canonical mixed original inputs must identify two distinct actual images")


def project_bindings(row, case_dir):
    bindings = deepcopy(row.get("canonical_full_input_bindings"))
    staged = []
    if bindings is None:
        return None, staged
    for index, binding in enumerate(bindings):
        item = binding["source_image"]
        local = Path(str(item.get("local_path") or item["path"]))
        if not local.is_file() or _sha(local) != item["sha256"]:
            raise ValueError("Canonical original source image changed before staging")
        remote = case_dir + "/original_inputs/" + str(index) + "/" + local.name
        staged.append({"local": str(local), "remote": remote, "sha256": item["sha256"],
                       "role": "canonical_original_input_" + str(index)})
        binding["source_image"] = {"path": remote, "sha256": item["sha256"]}
    return bindings, staged


def prepare_full_feeds(config, load_image):
    """Call the existing normal image loader exactly once per input, untimed."""
    sources = passthrough_sources(config)
    bindings = config.get("canonical_full_input_bindings")
    if not sources:
        if bindings is not None:
            raise ValueError("Canonical full-input bindings require an actual passthrough")
        return None
    if not isinstance(bindings, list) or len(bindings) != 2 or not callable(load_image):
        raise ValueError("Actual image passthrough lacks bound canonical input preparation")
    closure = {item["path"]: item["sha256"] for item in config["runtime_source_files"]}
    result = []
    for binding in bindings:
        item = binding["source_image"]
        contract = binding["preprocessing_contract"]
        specs = binding["inputs"]
        first = _content(next(iter(specs.values())))
        if (binding.get("schema") != SCHEMA or set(specs) != sources
                or binding.get("producer_source") != config["quality_reference"]["request"]
                or closure.get(item["path"]) != item["sha256"] or _sha(item["path"]) != item["sha256"]
                or contract.get("task") != config["task"]
                or any(_content(value) != first for value in specs.values())):
            raise ValueError("Canonical original-input source/normal preprocessing differs")
        value = load_image(Path(item["path"]), target_hw=tuple(contract["target_hw"]),
            dtype=np.dtype(first["dtype"]), scale=contract["image_scale"],
            letterbox=contract["preprocess_mode"] == "letterbox", preprocessing_contract=contract)
        if value is None or _snapshot(value) != first:
            raise ValueError("Canonical original-input bytes differ from actual normal Quality; normalization is not interchangeable")
        owned = np.ascontiguousarray(value)
        owned.flags.writeable = False
        result.append({name: owned for name in sources})
    return result
