if not api_key:
    import new_flow.utils.llm as llm_module
    logger.error(
        f"[TOKEN DEBUG] llm_config id={id(llm_config)} "
        f"module llm_config id={id(llm_module.llm_config)} "
        f"same_object={llm_config is llm_module.llm_config} "
        f"token_repr={llm_config.token!r}"
    )
    raise RuntimeError(...)
