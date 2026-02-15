from onnx_splitpoint_tool.core_params import (
    REQUIRED_MINIMUM_CV_KEYS,
    REQUIRED_MINIMUM_LLM_KEYS,
    exposed_param_keys,
    gui_state_to_params_dict,
    params_dict_to_gui_state,
)


def test_exposed_params_cover_required_minimum_cv_llm() -> None:
    exposed = set(exposed_param_keys())
    assert exposed >= set(REQUIRED_MINIMUM_CV_KEYS)
    assert exposed >= set(REQUIRED_MINIMUM_LLM_KEYS)


def test_gui_state_mapping_has_defaults_and_roundtrip() -> None:
    mapped = gui_state_to_params_dict({}, {})
    # Core-critical keys should always be present via defaults.
    assert "topk" in mapped
    assert "ranking" in mapped
    assert "llm_enable" in mapped

    state = params_dict_to_gui_state(
        {
            "topk": "23",
            "ranking": "latency",
            "llm_enable": True,
            "llm_preset": "Standard",
            "llm_mode": "decode",
        }
    )
    assert state["analysis"]["topk"] == "23"
    assert state["analysis"]["rank"] == "latency"
    assert state["llm"]["enable"] is True
    assert state["llm"]["preset"] == "Standard"
