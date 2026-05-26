def get_vlm_model(config):

    vlm_name = config.framework.qwenvl.base_vlm
    framework_name = config.framework.get("name", "") if hasattr(config.framework, "get") else getattr(config.framework, "name", "")
    interface_variant = config.framework.qwenvl.get("interface_variant", None) if hasattr(config.framework.qwenvl, "get") else getattr(config.framework.qwenvl, "interface_variant", None)

    if framework_name == "QwenPaDTPI" or interface_variant == "padt":
        from .QWen2_5_PaDT import _QWen_PaDT_VL_Interface
        return _QWen_PaDT_VL_Interface(config)

    if "Qwen2.5-VL" in vlm_name or "nora" in vlm_name.lower(): # temp for some ckpt
        from .QWen2_5 import _QWen_VL_Interface
        return _QWen_VL_Interface(config)
    elif "Qwen3-VL" in vlm_name:
        from .QWen3 import _QWen3_VL_Interface
        return _QWen3_VL_Interface(config)
    elif "Qwen3.5" in vlm_name:
        from .QWen3_5 import _QWen3_5_VL_Interface
        return _QWen3_5_VL_Interface(config)
    elif "florence" in vlm_name.lower(): # temp for some ckpt
        from .Florence2 import _Florence_Interface
        return _Florence_Interface(config)
    elif "cosmos-reason2" in vlm_name.lower():
        from .CosmosReason2 import _CosmosReason2_Interface
        return _CosmosReason2_Interface(config)
    else:
        raise NotImplementedError(f"VLM model {vlm_name} not implemented")
