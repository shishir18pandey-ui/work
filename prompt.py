 • Run: crewai traces enable                                                 │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

2026-08-27 13:02:35,288 - new_flow.utils.llm - ERROR - === FULL EXCEPTION TRACEBACK ===
2026-08-27 13:02:35,293 - new_flow.utils.llm - ERROR - Traceback (most recent call last):
  File "/usr/local/lib/python3.11/site-packages/litellm/llms/openai/openai.py", line 823, in acompletion
    headers, response = await self.make_openai_chat_completion_request(
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/litellm/litellm_core_utils/logging_utils.py", line 190, in async_wrapper
    result = await func(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/litellm/llms/openai/openai.py", line 454, in make_openai_chat_completion_request
    raise e
  File "/usr/local/lib/python3.11/site-packages/litellm/llms/openai/openai.py", line 436, in make_openai_chat_completion_request
    await openai_aclient.chat.completions.with_raw_response.create(
  File "/usr/local/lib/python3.11/site-packages/openai/_legacy_response.py", line 386, in wrapped
    return cast(LegacyAPIResponse[R], await func(*args, **kwargs))
                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/openai/resources/chat/completions/completions.py", line 2907, in create
    return await self._post(
           ^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/openai/_base_client.py", line 1992, in post
    return await self.request(cast_to, opts, stream=stream, stream_cls=stream_cls)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/openai/_base_client.py", line 1777, in request
    raise self._make_status_error_from_response(err.response) from None
openai.BadRequestError: Error code: 400 - {'error': {'message': "litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError: Hosted_vllmException - This model's maximum context length is 150000 tokens. However, you requested 0 output tokens and your prompt contains at least 150001 input tokens, for a total of at least 150001 tokens. Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_tokens, value=150001)\nmodel=/app/models/MiniMax-M2.5. context_window_fallbacks=None. fallbacks=None.\n\nSet 'context_window_fallback' - https://docs.litellm.ai/docs/routing#fallbacks. Received Model Group=/app/models/MiniMax-M2.5\nAvailable Model Group Fallbacks=None", 'type': None, 'param': None, 'code': '400'}}

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/local/lib/python3.11/site-packages/litellm/main.py", line 599, in acompletion
    response = await init_response
               ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/litellm/llms/openai/openai.py", line 870, in acompletion
    raise OpenAIError(
litellm.llms.openai.common_utils.OpenAIError: Error code: 400 - {'error': {'message': "litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError: Hosted_vllmException - This model's maximum context length is 150000 tokens. However, you requested 0 output tokens and your prompt contains at least 150001 input tokens, for a total of at least 150001 tokens. Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_tokens, value=150001)\nmodel=/app/models/MiniMax-M2.5. context_window_fallbacks=None. fallbacks=None.\n\nSet 'context_window_fallback' - https://docs.litellm.ai/docs/routing#fallbacks. Received Model Group=/app/models/MiniMax-M2.5\nAvailable Model Group Fallbacks=None", 'type': None, 'param': None, 'code': '400'}}

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/local/lib/python3.11/site-packages/crewai/llm.py", line 1339, in _ahandle_non_streaming_response
    response = await litellm.acompletion(**params)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/utils/minimax_tool_call_patch.py", line 65, in acompletion
    return _recover(await _orig_acompletion(*args, **kwargs))
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/litellm/utils.py", line 1642, in wrapper_async
    raise e
  File "/usr/local/lib/python3.11/site-packages/litellm/utils.py", line 1488, in wrapper_async
    result = await original_function(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/litellm/main.py", line 618, in acompletion
    raise exception_type(
          ^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/litellm/litellm_core_utils/exception_mapping_utils.py", line 2328, in exception_type
    raise e
  File "/usr/local/lib/python3.11/site-packages/litellm/litellm_core_utils/exception_mapping_utils.py", line 363, in exception_type
    raise ContextWindowExceededError(
litellm.exceptions.ContextWindowExceededError: litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError: OpenAIException - litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError: Hosted_vllmException - This model's maximum context length is 150000 tokens. However, you requested 0 output tokens and your prompt contains at least 150001 input tokens, for a total of at least 150001 tokens. Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_tokens, value=150001)
model=/app/models/MiniMax-M2.5. context_window_fallbacks=None. fallbacks=None.

Set 'context_window_fallback' - https://docs.litellm.ai/docs/routing#fallbacks. Received Model Group=/app/models/MiniMax-M2.5
Available Model Group Fallbacks=None

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/app/new_flow/utils/llm.py", line 96, in run_crew_with_retry_async
    result = await crew_coro
             ^^^^^^^^^^^^^^^
  File "/app/new_flow/agents/execute_agent_jaeger.py", line 734, in run_jaeger_only_async
    return await agent.execute(
           ^^^^^^^^^^^^^^^^^^^^
  File "/app/new_flow/agents/execute_agent_jaeger.py", line 190, in execute
    decision = await self._decide_after_traces(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/new_flow/agents/execute_agent_jaeger.py", line 534, in _decide_after_traces
    result = await crew.akickoff()
             ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/crew.py", line 976, in akickoff
    result = await self._arun_sequential_process()
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/crew.py", line 1035, in _arun_sequential_process
    return await self._aexecute_tasks(self.tasks)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/crew.py", line 1097, in _aexecute_tasks
    task_output = await task.aexecute_sync(
                  ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/task.py", line 578, in aexecute_sync
    return await self._aexecute_core(agent, context, tools)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/task.py", line 694, in _aexecute_core
    raise e  # Re-raise the exception after emitting the event
    ^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/task.py", line 602, in _aexecute_core
    result = await agent.aexecute_task(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/agent/core.py", line 892, in aexecute_task
    result = await self._handle_execution_error_async(e, task, context, tools)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/agent/core.py", line 698, in _handle_execution_error_async
    return await self.aexecute_task(task, context, tools)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/agent/core.py", line 892, in aexecute_task
    result = await self._handle_execution_error_async(e, task, context, tools)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/agent/core.py", line 698, in _handle_execution_error_async
    return await self.aexecute_task(task, context, tools)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/agent/core.py", line 892, in aexecute_task
    result = await self._handle_execution_error_async(e, task, context, tools)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/agent/core.py", line 697, in _handle_execution_error_async
    self._check_execution_error(e, task)
  File "/usr/local/lib/python3.11/site-packages/crewai/agent/core.py", line 656, in _check_execution_error
    raise e
  File "/usr/local/lib/python3.11/site-packages/crewai/agent/core.py", line 879, in aexecute_task
    result = await self._aexecute_without_timeout(task_prompt, task)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/agent/core.py", line 937, in _aexecute_without_timeout
    result = await self.agent_executor.ainvoke(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/agents/crew_agent_executor.py", line 1131, in ainvoke
    formatted_answer = await self._ainvoke_loop()
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/agents/crew_agent_executor.py", line 1170, in _ainvoke_loop
    return await self._ainvoke_loop_react()
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/agents/crew_agent_executor.py", line 1287, in _ainvoke_loop_react
    handle_context_length(
  File "/usr/local/lib/python3.11/site-packages/crewai/utilities/agent_utils.py", line 712, in handle_context_length
    summarize_messages(
  File "/usr/local/lib/python3.11/site-packages/crewai/utilities/agent_utils.py", line 980, in summarize_messages
    summarized_contents = pool.submit(ctx.run, asyncio.run, coro).result()
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/concurrent/futures/_base.py", line 456, in result
    return self.__get_result()
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/concurrent/futures/_base.py", line 401, in __get_result
    raise self._exception
  File "/usr/local/lib/python3.11/concurrent/futures/thread.py", line 58, in run
    result = self.fn(*self.args, **self.kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/asyncio/runners.py", line 190, in run
    return runner.run(main)
           ^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/asyncio/runners.py", line 118, in run
    return self._loop.run_until_complete(task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/asyncio/base_events.py", line 653, in run_until_complete
    return future.result()
           ^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/utilities/agent_utils.py", line 896, in _asummarize_chunks
    results = await asyncio.gather(*[_summarize_one(chunk) for chunk in chunks])
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/utilities/agent_utils.py", line 892, in _summarize_one
    summary = await llm.acall(summarization_messages, callbacks=callbacks)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/llm.py", line 1931, in acall
    return await self._ahandle_non_streaming_response(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/llm.py", line 1356, in _ahandle_non_streaming_response
    raise LLMContextLengthExceededError(error_msg) from e
crewai.utilities.exceptions.context_window_exceeding_exception.LLMContextLengthExceededError: LLM context length exceeded. Original error: litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError: OpenAIException - litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError: Hosted_vllmException - This model's maximum context length is 150000 tokens. However, you requested 0 output tokens and your prompt contains at least 150001 input tokens, for a total of at least 150001 tokens. Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_tokens, value=150001)
model=/app/models/MiniMax-M2.5. context_window_fallbacks=None. fallbacks=None.

Set 'context_window_fallback' - https://docs.litellm.ai/docs/routing#fallbacks. Received Model Group=/app/models/MiniMax-M2.5
Available Model Group Fallbacks=None
Consider using a smaller input or implementing a text splitting strategy.

2026-08-27 13:02:35,293 - new_flow.utils.llm - ERROR - === EXCEPTION TYPE: LLMContextLengthExceededError ===
2026-08-27 13:02:35,293 - new_flow.utils.llm - ERROR - === EXCEPTION MESSAGE: LLM context length exceeded. Original error: litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError: OpenAIException - litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError: Hosted_vllmException - This model's maximum context length is 150000 tokens. However, you requested 0 output tokens and your prompt contains at least 150001 input tokens, for a total of at least 150001 tokens. Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_tokens, value=150001)
model=/app/models/MiniMax-M2.5. context_window_fallbacks=None. fallbacks=None.

Set 'context_window_fallback' - https://docs.litellm.ai/docs/routing#fallbacks. Received Model Group=/app/models/MiniMax-M2.5
Available Model Group Fallbacks=None
Consider using a smaller input or implementing a text splitting strategy. ===
[CrewAIEventsBus] Warning: Event pairing mismatch. 'method_execution_failed' 
closed 'agent_execution_started' (expected 'method_execution_started')
2026-08-27 13:02:35,302 - crewai.flow.flow - ERROR - Error executing listener run_resolver_crew: LLM context length exceeded. Original error: litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError: OpenAIException - litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError: Hosted_vllmException - This model's maximum context length is 150000 tokens. However, you requested 0 output tokens and your prompt contains at least 150001 input tokens, for a total of at least 150001 tokens. Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_tokens, value=150001)
model=/app/models/MiniMax-M2.5. context_window_fallbacks=None. fallbacks=None.

Set 'context_window_fallback' - https://docs.litellm.ai/docs/routing#fallbacks. Received Model Group=/app/models/MiniMax-M2.5
Available Model Group Fallbacks=None
Consider using a smaller input or implementing a text splitting strategy.
2026-08-27 13:02:35,302 - __main__ - ERROR - ✗ Flow FAILED | incident=028b58983bc38710b6986f34c3e45a87 module=new_flow.flow error=LLM context length exceeded. Original error: litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError: OpenAIException - litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError: Hosted_vllmException - This model's maximum context length is 150000 tokens. However, you requested 0 output tokens and your prompt contains at least 150001 input tokens, for a total of at least 150001 tokens. Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_tokens, value=150001)
model=/app/models/MiniMax-M2.5. context_window_fallbacks=None. fallbacks=None.

Set 'context_window_fallback' - https://docs.litellm.ai/docs/routing#fallbacks. Received Model Group=/app/models/MiniMax-M2.5
Available Model Group Fallbacks=None
Consider using a smaller input or implementing a text splitting strategy.
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/site-packages/litellm/llms/openai/openai.py", line 823, in acompletion
    headers, response = await self.make_openai_chat_completion_request(
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/litellm/litellm_core_utils/logging_utils.py", line 190, in async_wrapper
    result = await func(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/litellm/llms/openai/openai.py", line 454, in make_openai_chat_completion_request
    raise e
  File "/usr/local/lib/python3.11/site-packages/litellm/llms/openai/openai.py", line 436, in make_openai_chat_completion_request
    await openai_aclient.chat.completions.with_raw_response.create(
  File "/usr/local/lib/python3.11/site-packages/openai/_legacy_response.py", line 386, in wrapped
    return cast(LegacyAPIResponse[R], await func(*args, **kwargs))
                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/openai/resources/chat/completions/completions.py", line 2907, in create
    return await self._post(
           ^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/openai/_base_client.py", line 1992, in post
    return await self.request(cast_to, opts, stream=stream, stream_cls=stream_cls)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/openai/_base_client.py", line 1777, in request
    raise self._make_status_error_from_response(err.response) from None
openai.BadRequestError: Error code: 400 - {'error': {'message': "litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError: Hosted_vllmException - This model's maximum context length is 150000 tokens. However, you requested 0 output tokens and your prompt contains at least 150001 input tokens, for a total of at least 150001 tokens. Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_tokens, value=150001)\nmodel=/app/models/MiniMax-M2.5. context_window_fallbacks=None. fallbacks=None.\n\nSet 'context_window_fallback' - https://docs.litellm.ai/docs/routing#fallbacks. Received Model Group=/app/models/MiniMax-M2.5\nAvailable Model Group Fallbacks=None", 'type': None, 'param': None, 'code': '400'}}

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/local/lib/python3.11/site-packages/litellm/main.py", line 599, in acompletion
    response = await init_response
               ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/litellm/llms/openai/openai.py", line 870, in acompletion
    raise OpenAIError(
litellm.llms.openai.common_utils.OpenAIError: Error code: 400 - {'error': {'message': "litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError: Hosted_vllmException - This model's maximum context length is 150000 tokens. However, you requested 0 output tokens and your prompt contains at least 150001 input tokens, for a total of at least 150001 tokens. Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_tokens, value=150001)\nmodel=/app/models/MiniMax-M2.5. context_window_fallbacks=None. fallbacks=None.\n\nSet 'context_window_fallback' - https://docs.litellm.ai/docs/routing#fallbacks. Received Model Group=/app/models/MiniMax-M2.5\nAvailable Model Group Fallbacks=None", 'type': None, 'param': None, 'code': '400'}}

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/local/lib/python3.11/site-packages/crewai/llm.py", line 1339, in _ahandle_non_streaming_response
    response = await litellm.acompletion(**params)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/utils/minimax_tool_call_patch.py", line 65, in acompletion
    return _recover(await _orig_acompletion(*args, **kwargs))
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/litellm/utils.py", line 1642, in wrapper_async
    raise e
  File "/usr/local/lib/python3.11/site-packages/litellm/utils.py", line 1488, in wrapper_async
    result = await original_function(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/litellm/main.py", line 618, in acompletion
    raise exception_type(
          ^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/litellm/litellm_core_utils/exception_mapping_utils.py", line 2328, in exception_type
    raise e
  File "/usr/local/lib/python3.11/site-packages/litellm/litellm_core_utils/exception_mapping_utils.py", line 363, in exception_type
    raise ContextWindowExceededError(
litellm.exceptions.ContextWindowExceededError: litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError: OpenAIException - litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError: Hosted_vllmException - This model's maximum context length is 150000 tokens. However, you requested 0 output tokens and your prompt contains at least 150001 input tokens, for a total of at least 150001 tokens. Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_tokens, value=150001)
model=/app/models/MiniMax-M2.5. context_window_fallbacks=None. fallbacks=None.

Set 'context_window_fallback' - https://docs.litellm.ai/docs/routing#fallbacks. Received Model Group=/app/models/MiniMax-M2.5
Available Model Group Fallbacks=None

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/app/worker.py", line 138, in process_message
    await flow.akickoff()
  File "/usr/local/lib/python3.11/site-packages/crewai/flow/flow.py", line 2211, in akickoff
    return await self.kickoff_async(inputs, input_files)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/flow/flow.py", line 2091, in kickoff_async
    await asyncio.gather(*tasks)
  File "/usr/local/lib/python3.11/site-packages/crewai/flow/flow.py", line 2262, in _execute_start_method
    await self._execute_listeners(start_method_name, result, finished_event_id)
  File "/usr/local/lib/python3.11/site-packages/crewai/flow/flow.py", line 2552, in _execute_listeners
    await asyncio.gather(*tasks)
  File "/usr/local/lib/python3.11/site-packages/crewai/flow/flow.py", line 2820, in _execute_single_listener
    await self._execute_listeners(
  File "/usr/local/lib/python3.11/site-packages/crewai/flow/flow.py", line 2552, in _execute_listeners
    await asyncio.gather(*tasks)
  File "/usr/local/lib/python3.11/site-packages/crewai/flow/flow.py", line 2806, in _execute_single_listener
    listener_result, finished_event_id = await self._execute_method(
                                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/flow/flow.py", line 2428, in _execute_method
    raise e
  File "/usr/local/lib/python3.11/site-packages/crewai/flow/flow.py", line 2341, in _execute_method
    result = await method(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/new_flow/flow.py", line 351, in run_resolver_crew
    return await self._run_agentic_resolver(app)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/new_flow/flow.py", line 413, in _run_agentic_resolver
    execution_result = await run_crew_with_retry_async(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/new_flow/utils/llm.py", line 96, in run_crew_with_retry_async
    result = await crew_coro
             ^^^^^^^^^^^^^^^
  File "/app/new_flow/agents/execute_agent_jaeger.py", line 734, in run_jaeger_only_async
    return await agent.execute(
           ^^^^^^^^^^^^^^^^^^^^
  File "/app/new_flow/agents/execute_agent_jaeger.py", line 190, in execute
    decision = await self._decide_after_traces(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/new_flow/agents/execute_agent_jaeger.py", line 534, in _decide_after_traces
    result = await crew.akickoff()
             ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/crew.py", line 976, in akickoff
    result = await self._arun_sequential_process()
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/crew.py", line 1035, in _arun_sequential_process
    return await self._aexecute_tasks(self.tasks)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/crew.py", line 1097, in _aexecute_tasks
    task_output = await task.aexecute_sync(
                  ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/task.py", line 578, in aexecute_sync
    return await self._aexecute_core(agent, context, tools)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/task.py", line 694, in _aexecute_core
    raise e  # Re-raise the exception after emitting the event
    ^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/task.py", line 602, in _aexecute_core
    result = await agent.aexecute_task(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/agent/core.py", line 892, in aexecute_task
    result = await self._handle_execution_error_async(e, task, context, tools)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/agent/core.py", line 698, in _handle_execution_error_async
    return await self.aexecute_task(task, context, tools)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/agent/core.py", line 892, in aexecute_task
    result = await self._handle_execution_error_async(e, task, context, tools)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/agent/core.py", line 698, in _handle_execution_error_async
    return await self.aexecute_task(task, context, tools)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/agent/core.py", line 892, in aexecute_task
    result = await self._handle_execution_error_async(e, task, context, tools)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/agent/core.py", line 697, in _handle_execution_error_async
    self._check_execution_error(e, task)
  File "/usr/local/lib/python3.11/site-packages/crewai/agent/core.py", line 656, in _check_execution_error
    raise e
  File "/usr/local/lib/python3.11/site-packages/crewai/agent/core.py", line 879, in aexecute_task
    result = await self._aexecute_without_timeout(task_prompt, task)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/agent/core.py", line 937, in _aexecute_without_timeout
    result = await self.agent_executor.ainvoke(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/agents/crew_agent_executor.py", line 1131, in ainvoke
    formatted_answer = await self._ainvoke_loop()
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/agents/crew_agent_executor.py", line 1170, in _ainvoke_loop
    return await self._ainvoke_loop_react()
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/agents/crew_agent_executor.py", line 1287, in _ainvoke_loop_react
    handle_context_length(
  File "/usr/local/lib/python3.11/site-packages/crewai/utilities/agent_utils.py", line 712, in handle_context_length
    summarize_messages(
  File "/usr/local/lib/python3.11/site-packages/crewai/utilities/agent_utils.py", line 980, in summarize_messages
    summarized_contents = pool.submit(ctx.run, asyncio.run, coro).result()
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/concurrent/futures/_base.py", line 456, in result
    return self.__get_result()
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/concurrent/futures/_base.py", line 401, in __get_result
    raise self._exception
  File "/usr/local/lib/python3.11/concurrent/futures/thread.py", line 58, in run
    result = self.fn(*self.args, **self.kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/asyncio/runners.py", line 190, in run
    return runner.run(main)
           ^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/asyncio/runners.py", line 118, in run
    return self._loop.run_until_complete(task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/asyncio/base_events.py", line 653, in run_until_complete
    return future.result()
           ^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/utilities/agent_utils.py", line 896, in _asummarize_chunks
    results = await asyncio.gather(*[_summarize_one(chunk) for chunk in chunks])
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/utilities/agent_utils.py", line 892, in _summarize_one
    summary = await llm.acall(summarization_messages, callbacks=callbacks)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/llm.py", line 1931, in acall
    return await self._ahandle_non_streaming_response(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/llm.py", line 1356, in _ahandle_non_streaming_response
    raise LLMContextLengthExceededError(error_msg) from e
crewai.utilities.exceptions.context_window_exceeding_exception.LLMContextLengthExceededError: LLM context length exceeded. Original error: litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError: OpenAIException - litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError: Hosted_vllmException - This model's maximum context length is 150000 tokens. However, you requested 0 output tokens and your prompt contains at least 150001 input tokens, for a total of at least 150001 tokens. Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_tokens, value=150001)
model=/app/models/MiniMax-M2.5. context_window_fallbacks=None. fallbacks=None.

Set 'context_window_fallback' - https://docs.litellm.ai/docs/routing#fallbacks. Received Model Group=/app/models/MiniMax-M2.5
Available Model Group Fallbacks=None
Consider using a smaller input or implementing a text splitting strategy.
2026-08-27 13:02:35,325 - utils.pii_masking_integration - INFO - PII masking output complete for ServiceNow
2026-08-27 13:02:36,421 - __main__ - INFO -   Fallback rejection sent | incident=028b58983bc38710b6986f34c3e45a87
2026-08-27 13:05:26,568 - new_flow.utils.llm - INFO - OPENAI_API_KEY refreshed
2026-08-27 13:15:26,647 - new_flow.utils.llm - INFO - OPENAI_API_KEY refreshed
shishir.pandey_tho@0325LTPB0124444 ~ % 
