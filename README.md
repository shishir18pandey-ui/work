2026-08-18 20:40:15,397 - __main__ - INFO -   Group    : gen-ai-de-incident-managers
2026-08-18 20:40:15,397 - __main__ - INFO -   Broker   : b-1.dcawscentraluatkafka0.fil3x5.c4.kafka.ap-south-1.amazonaws.com:9096,b-2.dcawscentraluatkafka0.fil3x5.c4.kafka.ap-south-1.amazonaws.com:9096,b-3.dcawscentraluatkafka0.fil3x5.c4.kafka.ap-south-1.amazonaws.com:9096
2026-08-18 20:40:15,465 - new_flow.utils.llm - INFO - OPENAI_API_KEY refreshed
2026-08-18 20:42:04,254 - __main__ - INFO - → Message received | incident=3800011 event=new_incident partition=0 offset=486
2026-08-18 20:42:04,254 - __main__ - INFO - ✓ Offset committed | incident=3800011
2026-08-18 20:42:04,270 - __main__ - INFO - → Processing | incident=3800011 event=new_incident module=new_flow.flow
2026-08-18 20:42:04,390 - __main__ - INFO -   DB status | incident=3800011 status=in_progress
2026-08-18 20:42:05,945 - crewai.cli.config - INFO - Using config path: /root/.config/crewai/settings.json
2026-08-18 20:42:06,375 - new_flow.tools.service_metadata - INFO - Loaded service metadata for apps: ['optimus', 'cbs', 'idp']
2026-08-18 20:42:08,290 - __main__ - INFO -   Flow type | incident=3800011 type=new_incident
ot-collector.tracing.svc.cluster.local tetsing1
========== LLM CONFIG ==========
URL: https://llm-api.iservebetter.idfcfirstbank.com/minimax-m2-entauth/v1
TOKEN: srq88A8vCstapjBgSZHtd_EPzTQ-UaUp1L9vboxYV48.IWkdFmA8KihkDhPPn8oSyDlACnfipNdzM7kxweGdZ-0
================================
╭───────────────────────────── 🌊 Flow Execution ──────────────────────────────╮
│                                                                              │
│  Starting Flow Execution                                                     │
│  Name: IncidentManagementFlow                                                │
│  ID: 8d00afe9-a416-433a-a16c-b5f535a0eadd                                    │
│                                                                              │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

╭────────────────────────────── 🌊 Flow Started ───────────────────────────────╮
│                                                                              │
│  Flow Started                                                                │
│  Name: IncidentManagementFlow                                                │
│  ID: 8d00afe9-a416-433a-a16c-b5f535a0eadd                                    │
│                                                                              │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

Flow started with ID: 8d00afe9-a416-433a-a16c-b5f535a0eadd
2026-08-18 20:42:24,327 - crewai.flow.flow - INFO - Flow started with ID: 8d00afe9-a416-433a-a16c-b5f535a0eadd
╭─────────────────────────── 🔄 Flow Method Running ───────────────────────────╮
│                                                                              │
│  Method: initialize_and_classify                                             │
│  Status: Running                                                             │
│                                                                              │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

Generated incident description for UCIC 1056011
Using CA certificate: ./IDFCBANKCA.pem
=== Intent Classifier LLM ===
Model: openai//app/models/Qwen3-14B-FP8
Base URL: https://llm-api.iservebetter.idfcfirstbank.com/qwen3-14b/v1
CA certificate: ./IDFCBANKCA.pem
==============================
╭───────────────────────── 🚀 Crew Execution Started ──────────────────────────╮
│                                                                              │
│  Crew Execution Started                                                      │
│  Name: crew                                                                  │
│  ID: 8fbbbee9-8577-4fcd-a410-0f4276b1e8c7                                    │
│                                                                              │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

20:42:40 - LiteLLM:INFO: utils.py:3427 - 
LiteLLM completion() model= /app/models/Qwen3-14B-FP8; provider = openai
2026-08-18 20:42:40,391 - LiteLLM - INFO - 
LiteLLM completion() model= /app/models/Qwen3-14B-FP8; provider = openai
╭────────────────────────────── 📋 Task Started ───────────────────────────────╮
│                                                                              │
│  Task Started                                                                │
│  Name: Analyze the user input and categorize it with detailed analysis.      │
│                                                                              │
│  Interaction History:                                                        │
│  ```                                                                         │
│  ['NA']                                                                      │
│  ```                                                                         │
│                                                                              │
│  User Input:                                                                 │
│  ```                                                                         │
│  Short Description: <script>alert('XSS')</script> Customer is unable to do   │
│  transfer of money                                                           │
│  Description: Customer is unable to do transfer of money                     │
│  ```                                                                         │
│                                                                              │
│  Latest Interaction:                                                         │
│  ```                                                                         │
│  NA                                                                          │
│  ```                                                                         │
│                                                                              │
│  Depending on history you can figure out the latest question if it exists.   │
│  Categorise intent on the basis of `Latest Interaction`. If `Latest          │
│  Interaction` is NA, classify on the basis of `User Input`.                  │
│                                                                              │
│  **Categories**:                                                             │
│                                                                              │
│  - **closure**: Greeting, thanks, or ending the chat.                        │
│                                                                              │
│  - **rebuttal**: User is disagreeing, correcting the system, or insisting    │
│  that information they previously provided is correct (e.g. 'I already told  │
│  you', 'That's wrong', 'This is correct.').                                  │
│                                                                              │
│  - **additional_info**: Providing IDs, account numbers or subsequent         │
│  question/information asked.                                                 │
│                                                                              │
│  **Rule**: If the User Input contradicts the Latest Interaction or           │
│  expresses frustration with the system's request, it MUST be 'rebuttal'.     │
│                                                                              │
│  **Analysis Required**:                                                      │
│  Based on the user input and history, provide:                               │
│  1. `user_goal`: What the user is trying to achieve or accomplish.           │
│  2. `issue_description`: The specific problem or issue they're facing.       │
│  3. `problem_summary`: A detailed write-up explaining the issue in context.  │
│                                                                              │
│  Output your response as valid JSON with the following structure:            │
│  {                                                                           │
│    "intent": "<category>",                                                   │
│    "user_goal": "<what user wants to accomplish>",                           │
│    "issue_description": "<specific problem>",                                │
│    "problem_summary": "<detailed write-up>"                                  │
│  }                                                                           │
│  Do not add any text before or after the JSON.                               │
│  ID: ec31c4f6-ff31-4d6a-8d90-2f1b34e6bd45                                    │
│                                                                              │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
╭────────────────────────────── 🤖 Agent Started ──────────────────────────────╮
│                                                                              │
│  Agent: Intent Classifier                                                    │
│                                                                              │
│  Task: Analyze the user input and categorize it with detailed analysis.      │
│                                                                              │
│  Interaction History:                                                        │
│  ```                                                                         │
│  ['NA']                                                                      │
│  ```                                                                         │
│                                                                              │
│  User Input:                                                                 │
│  ```                                                                         │
│  Short Description: <script>alert('XSS')</script> Customer is unable to do   │
│  transfer of money                                                           │
│  Description: Customer is unable to do transfer of money                     │
│  ```                                                                         │
│                                                                              │
│  Latest Interaction:                                                         │
│  ```                                                                         │
│  NA                                                                          │
│  ```                                                                         │
│                                                                              │
│  Depending on history you can figure out the latest question if it exists.   │
│  Categorise intent on the basis of `Latest Interaction`. If `Latest          │
│  Interaction` is NA, classify on the basis of `User Input`.                  │
│                                                                              │
│  **Categories**:                                                             │
│                                                                              │
│  - **closure**: Greeting, thanks, or ending the chat.                        │
│                                                                              │
│  - **rebuttal**: User is disagreeing, correcting the system, or insisting    │
│  that information they previously provided is correct (e.g. 'I already told  │
│  you', 'That's wrong', 'This is correct.').                                  │
│                                                                              │
│  - **additional_info**: Providing IDs, account numbers or subsequent         │
│  question/information asked.                                                 │
│                                                                              │
│  **Rule**: If the User Input contradicts the Latest Interaction or           │
│  expresses frustration with the system's request, it MUST be 'rebuttal'.     │
│                                                                              │
│  **Analysis Required**:                                                      │
│  Based on the user input and history, provide:                               │
│  1. `user_goal`: What the user is trying to achieve or accomplish.           │
│  2. `issue_description`: The specific problem or issue they're facing.       │
│  3. `problem_summary`: A detailed write-up explaining the issue in context.  │
│                                                                              │
│  Output your response as valid JSON with the following structure:            │
│  {                                                                           │
│    "intent": "<category>",                                                   │
│    "user_goal": "<what user wants to accomplish>",                           │
│    "issue_description": "<specific problem>",                                │
│    "problem_summary": "<detailed write-up>"                                  │
│  }                                                                           │
│  Do not add any text before or after the JSON.                               │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯


2026-08-18 20:42:40,700 - openai._base_client - INFO - Retrying request to /chat/completions in 0.460699 seconds
2026-08-18 20:42:41,167 - openai._base_client - INFO - Retrying request to /chat/completions in 0.852374 seconds
╭──────────────────────────────── ❌ LLM Error ────────────────────────────────╮
│                                                                              │
│  LLM Call Failed                                                             │
│  Error: litellm.InternalServerError: InternalServerError: OpenAIException -  │
│  An unexpected error occurred                                                │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

╭────────────────────────────── 📋 Task Failure ───────────────────────────────╮
│                                                                              │
│  Task Failed                                                                 │
│  Name: Analyze the user input and categorize it with detailed analysis.      │
│                                                                              │
│  Interaction History:                                                        │
│  ```                                                                         │
│  ['NA']                                                                      │
│  ```                                                                         │
│                                                                              │
│  User Input:                                                                 │
│  ```                                                                         │
│  Short Description: <script>alert('XSS')</script> Customer is unable to do   │
│  transfer of money                                                           │
│  Description: Customer is unable to do transfer of money                     │
│  ```                                                                         │
│                                                                              │
│  Latest Interaction:                                                         │
│  ```                                                                         │
│  NA                                                                          │
│  ```                                                                         │
│                                                                              │
│  Depending on history you can figure out the latest question if it exists.   │
│  Categorise intent on the basis of `Latest Interaction`. If `Latest          │
│  Interaction` is NA, classify on the basis of `User Input`.                  │
│                                                                              │
│  **Categories**:                                                             │
│                                                                              │
│  - **closure**: Greeting, thanks, or ending the chat.                        │
│                                                                              │
│  - **rebuttal**: User is disagreeing, correcting the system, or insisting    │
│  that information they previously provided is correct (e.g. 'I already told  │
│  you', 'That's wrong', 'This is correct.').                                  │
│                                                                              │
│  - **additional_info**: Providing IDs, account numbers or subsequent         │
│  question/information asked.                                                 │
│                                                                              │
│  **Rule**: If the User Input contradicts the Latest Interaction or           │
│  expresses frustration with the system's request, it MUST be 'rebuttal'.     │
│                                                                              │
│  **Analysis Required**:                                                      │
│  Based on the user input and history, provide:                               │
│  1. `user_goal`: What the user is trying to achieve or accomplish.           │
│  2. `issue_description`: The specific problem or issue they're facing.       │
│  3. `problem_summary`: A detailed write-up explaining the issue in context.  │
│                                                                              │
│  Output your response as valid JSON with the following structure:            │
│  {                                                                           │
│    "intent": "<category>",                                                   │
│    "user_goal": "<what user wants to accomplish>",                           │
│    "issue_description": "<specific problem>",                                │
│    "problem_summary": "<detailed write-up>"                                  │
│  }                                                                           │
│  Do not add any text before or after the JSON.                               │
│  Agent: Intent Classifier                                                    │
│                                                                              │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯



╭────────────────────────── Tracing Preference Saved ──────────────────────────╮
│                                                                              │
│  Info: Tracing has been disabled.                                            │
│                                                                              │
│  Your preference has been saved. Future Crew/Flow executions will not        │
│  collect traces.                                                             │
│                                                                              │
│  To enable tracing later, do any one of these:                               │
│  • Set tracing=True in your Crew/Flow code                                   │
│  • Set CREWAI_TRACING_ENABLED=true in your project's .env file               │
│  • Run: crewai traces enable                                                 │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

2026-08-18 20:42:42,045 - new_flow.utils.llm - ERROR - === FULL EXCEPTION TRACEBACK ===
╭──────────────────────────────── Crew Failure ────────────────────────────────╮
│                                                                              │
│  Crew Execution Failed                                                       │
│  Name: crew                                                                  │
│  ID: 8fbbbee9-8577-4fcd-a410-0f4276b1e8c7                                    │
│                                                                              │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

2026-08-18 20:42:42,053 - new_flow.utils.llm - ERROR - Traceback (most recent call last):
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
openai.InternalServerError: Error code: 500 - {'message': 'An unexpected error occurred', 'request_id': 'aab054656b8886024f27620c920f7404'}

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/local/lib/python3.11/site-packages/litellm/main.py", line 599, in acompletion
    response = await init_response
               ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/litellm/llms/openai/openai.py", line 870, in acompletion
    raise OpenAIError(
litellm.llms.openai.common_utils.OpenAIError: Error code: 500 - {'message': 'An unexpected error occurred', 'request_id': 'aab054656b8886024f27620c920f7404'}

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/app/new_flow/utils/llm.py", line 101, in run_crew_with_retry_async
    result = await crew_coro
             ^^^^^^^^^^^^^^^
  File "/app/new_flow/agents/intent_classifier.py", line 778, in run_classifier_with_enrichment_async
    intent_result = await run_intent_classifier_crew_async(
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/new_flow/agents/intent_classifier.py", line 651, in run_intent_classifier_crew_async
    result = await crew.akickoff(
             ^^^^^^^^^^^^^^^^^^^^
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
  File "/usr/local/lib/python3.11/site-packages/crewai/agent/core.py", line 697, in _handle_execution_error_async
    self._check_execution_error(e, task)
  File "/usr/local/lib/python3.11/site-packages/crewai/agent/core.py", line 643, in _check_execution_error
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
  File "/usr/local/lib/python3.11/site-packages/crewai/agents/crew_agent_executor.py", line 1285, in _ainvoke_loop_react
    raise e
  File "/usr/local/lib/python3.11/site-packages/crewai/agents/crew_agent_executor.py", line 1195, in _ainvoke_loop_react
    answer = await aget_llm_response(
             ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/utilities/agent_utils.py", line 531, in aget_llm_response
    raise e
  File "/usr/local/lib/python3.11/site-packages/crewai/utilities/agent_utils.py", line 521, in aget_llm_response
    answer = await llm.acall(
             ^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/llm.py", line 1931, in acall
    return await self._ahandle_non_streaming_response(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
  File "/usr/local/lib/python3.11/site-packages/litellm/litellm_core_utils/exception_mapping_utils.py", line 527, in exception_type
    raise InternalServerError(
litellm.exceptions.InternalServerError: litellm.InternalServerError: InternalServerError: OpenAIException - An unexpected error occurred

2026-08-18 20:42:42,053 - new_flow.utils.llm - ERROR - === EXCEPTION TYPE: InternalServerError ===
2026-08-18 20:42:42,053 - new_flow.utils.llm - ERROR - === EXCEPTION MESSAGE: litellm.InternalServerError: InternalServerError: OpenAIException - An unexpected error occurred ===
╭─────────────────────────── ❌ Flow Method Failed ────────────────────────────╮
│                                                                              │
│  Method: initialize_and_classify                                             │
│  Status: Failed                                                              │
│                                                                              │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

2026-08-18 20:42:42,057 - __main__ - ERROR - ✗ Flow FAILED | incident=3800011 module=new_flow.flow error=litellm.InternalServerError: InternalServerError: OpenAIException - An unexpected error occurred
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
openai.InternalServerError: Error code: 500 - {'message': 'An unexpected error occurred', 'request_id': 'aab054656b8886024f27620c920f7404'}

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/local/lib/python3.11/site-packages/litellm/main.py", line 599, in acompletion
    response = await init_response
               ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/litellm/llms/openai/openai.py", line 870, in acompletion
    raise OpenAIError(
litellm.llms.openai.common_utils.OpenAIError: Error code: 500 - {'message': 'An unexpected error occurred', 'request_id': 'aab054656b8886024f27620c920f7404'}

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/app/worker.py", line 138, in process_message
    await flow.akickoff()
  File "/usr/local/lib/python3.11/site-packages/crewai/flow/flow.py", line 2211, in akickoff
    return await self.kickoff_async(inputs, input_files)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/flow/flow.py", line 2091, in kickoff_async
    await asyncio.gather(*tasks)
  File "/usr/local/lib/python3.11/site-packages/crewai/flow/flow.py", line 2243, in _execute_start_method
    result, finished_event_id = await self._execute_method(
                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/flow/flow.py", line 2428, in _execute_method
    raise e
  File "/usr/local/lib/python3.11/site-packages/crewai/flow/flow.py", line 2341, in _execute_method
    result = await method(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/flow/flow.py", line 2288, in enhanced_method
    return await original_method(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/new_flow/flow.py", line 247, in initialize_and_classify
    classifier_output = await run_crew_with_retry_async(
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/new_flow/utils/llm.py", line 101, in run_crew_with_retry_async
    result = await crew_coro
             ^^^^^^^^^^^^^^^
  File "/app/new_flow/agents/intent_classifier.py", line 778, in run_classifier_with_enrichment_async
    intent_result = await run_intent_classifier_crew_async(
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/new_flow/agents/intent_classifier.py", line 651, in run_intent_classifier_crew_async
    result = await crew.akickoff(
             ^^^^^^^^^^^^^^^^^^^^
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
  File "/usr/local/lib/python3.11/site-packages/crewai/agent/core.py", line 697, in _handle_execution_error_async
    self._check_execution_error(e, task)
  File "/usr/local/lib/python3.11/site-packages/crewai/agent/core.py", line 643, in _check_execution_error
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
  File "/usr/local/lib/python3.11/site-packages/crewai/agents/crew_agent_executor.py", line 1285, in _ainvoke_loop_react
    raise e
  File "/usr/local/lib/python3.11/site-packages/crewai/agents/crew_agent_executor.py", line 1195, in _ainvoke_loop_react
    answer = await aget_llm_response(
             ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/utilities/agent_utils.py", line 531, in aget_llm_response
    raise e
  File "/usr/local/lib/python3.11/site-packages/crewai/utilities/agent_utils.py", line 521, in aget_llm_response
    answer = await llm.acall(
             ^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/llm.py", line 1931, in acall
    return await self._ahandle_non_streaming_response(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
  File "/usr/local/lib/python3.11/site-packages/litellm/litellm_core_utils/exception_mapping_utils.py", line 527, in exception_type
    raise InternalServerError(
litellm.exceptions.InternalServerError: litellm.InternalServerError: InternalServerError: OpenAIException - An unexpected error occurred
2026-08-18 20:42:43,022 - __main__ - INFO -   Fallback rejection sent | incident=3800011
shishir.pandey_tho@0325LTPB0124444 ~ % 
