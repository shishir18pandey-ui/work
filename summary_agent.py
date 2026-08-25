if historic_context and historic_context.strip() and historic_context != "No context found":
        return await run_context_only_summary_async(
            incident_description=incident_description,
            historic_context=historic_context,
            user_qa_pairs=user_qa_pairs
        )

    # ── Nothing worked at all: no logs, no historic match ──
    return SummaryOutput(
        diagnosis="BOT is unable to resolve, assign to an Engineer",
        solution="BOT is unable to resolve, assign to an Engineer",
        questions=[],
        resolved="no"
    )
