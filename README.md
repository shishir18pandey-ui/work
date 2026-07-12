Issue im facing is these










╭────────────────────────────── 🤖 Agent Started ──────────────────────────────╮
│                                                                              │
│  Agent: Incident Resolution Agent                                            │
│                                                                              │
│  Task: Extract customer identifiers from the incident.                       │
│                                                                              │
│  UCIC field: 1041338998                                                      │
│  Incident: Short Description: Customer is unable to do add funds             │
│  Description: Customer is unable to do add funds                             │
│                                                                              │
│  Extract these identifiers if present:                                       │
│  - ucic                                                                      │
│  - mobile_number                                                             │
│  - username                                                                  │
│  - customer_id                                                               │
│  - txn_id / txn_request_id                                                   │
│                                                                              │
│  Then use fetch_optimus_logs tool with:                                      │
│  - tag_name = the identifier type you found                                  │
│  - tag_value = the actual value                                              │
│  - service = pick from tool description based on incident type               │
│  - time_description = 'last 48 hours' unless user mentions a specific time   │
│                                                                              │
│  If no logs found with ucic, retry with mobile_number if available.          │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

10:08:49 - LiteLLM:INFO: utils.py:3427 - 
LiteLLM completion() model= /app/models/MiniMax-M2.5; provider = openai
2026-07-12 10:08:49,359 - LiteLLM - INFO - 
LiteLLM completion() model= /app/models/MiniMax-M2.5; provider = openai
╭─────────────────────── 🔧 Tool Execution Started (#1) ───────────────────────╮
│                                                                              │
│  Tool: payments_add_funds_check                                              │
│  Args: {"tag_name": "customer_id", "tag_value": "1041338998"}                │
│                                                                              │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

2026-07-12 10:08:51,058 - tools.tool - INFO - [JAEGER] service=payments-api customer_id=1041338998
2026-07-12 10:08:53,245 - tools.tool - INFO - [JAEGER] Done: scanned=38 failed=36
2026-07-12 10:08:53,259 - tools.tool - INFO - [JaegerTool payments_add_funds_check] found 38 traces using tag_name=customer_id
╭──────────────────────────────── Tool Output ─────────────────────────────────╮
│                                                                              │
│  {'service': 'payments-api', 'tag_name': 'customer_id', 'tag_value':         │
│  '1041338998', 'total_traces_scanned': 38, 'total_failed': 36,               │
│  'failed_traces': ['Time: 2026-07-10 15:18:06 IST\ndbd54ba61980a49c - nginx  │
│  - /api/beneficiary/v2/payee\nHTTP Status: 200\n    bf03de87aeceb264 -       │
│  beneficiary-api - /api/beneficiary/v2/payee\n    HTTP Status: 200\n         │
│  ERROR: Error while fetching the IP address from incoming request: client    │
│  ip empty/not present\n        88fe82a5e5ec3cb7 - beneficiary-api -          │
│  /api/beneficiary/v2/payee | request/response\n        HTTP Status: 200\n    │
│  request:                                                                    │
│  {"name":"Amu","nickName":"","type":"account","accountNumber":"123456789","  │
│  ifscCode":"SBIN0000001","bankName":"STATE BANK OF                           │
│  INDIA","transactionId":"c69ef816-6227-4969-817f-b514cae15336","transaction  │
│  Type":"ADD_PAYEE"}\n          response:                                     │
│  {"status":"success","payeeId":"150813"}\n        fbf8f71c5a33c1e4 -         │
│  beneficiary-api - /admin/oauth2/introspect | request/response\n             │
│  HTTP Status: 200\n          request:                                        │
│  --4ebf8aecbd9f9d5dcc8a5fac45d956e5c78a46d18b1717530c49301d0030\r\nContent-  │
│  Disposition: form-data;                                                     │
│  name="token"\r\n\r\nory_at_N1SZMXl15zdlrS35TB6cIWQyYhX5NDyW9IaKOoUbBek.2xv  │
│  tu5kwNhYBf9yLdG5o8Qym90vlps993_Gm5N0li5I\r\n--4ebf8aecbd9f9d5dcc8a5fac45d9  │
│  56e5c78a46d18b1717530c49301d0030--\r\n\n          response:                 │
│  {"active":true,"scope":"openid                                              │
│  offline","client_id":"ui-web","sub":"d9f81da9-68f6-4394-96a3-9379368ffd9e"  │
│  ,"exp":1783678583,"iat":1783676783,"nbf":1783676783,"aud":[],"iss":"https:  │
│  //app.uat-opt.idfcfirstbank.com/platform/oauth/","token_type":"Bearer","to  │
│  ken_use":"access_token"}\n\n        9a29cb7aeb598bb5 - beneficiary-api -    │
│  https://oauth-admin.uat-entauth.idfcfirstbank.com/admin/oauth2/introspect\  │
│  n        HTTP Status: 200\n            1c2a02e73c5e5555 - nginx -           │
│  /admin/oauth2/introspect\n            HTTP Status: 200\n                    │
│  c64fd151275d11d2 - beneficiary-api - /api/crypto/id-token/decrypt |         │
│  request/response\n        HTTP Status: 200\n          response: Response    │
│  body not logged for securit...                                              │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

10:08:53 - LiteLLM:INFO: utils.py:3427 - 
LiteLLM completion() model= /app/models/MiniMax-M2.5; provider = openai
2026-07-12 10:08:53,665 - LiteLLM - INFO - 
LiteLLM completion() model= /app/models/MiniMax-M2.5; provider = openai
╭────────────────────── ✅ Tool Execution Completed (#1) ──────────────────────╮
│                                                                              │
│  Tool Completed                                                              │
│  Tool: payments_add_funds_check                                              │
│  Output: {'service': 'payments-api', 'tag_name': 'customer_id',              │
│  'tag_value': '1041338998', 'total_traces_scanned': 38, 'total_failed': 36,  │
│  'failed_traces': ['Time: 2026-07-10 15:18:06 IST\ndbd54ba61980a49c - nginx  │
│  - /api/beneficiary/v2/payee\nHTTP Status: 200\n    bf03de87aeceb264 -       │
│  beneficiary-api - /api/beneficiary/v2/payee\n    HTTP Status: 200\n         │
│  ERROR: Error while fetching the IP address from incoming request: client    │
│  ip empty/not present\n        88fe82a5e5ec3cb7 - beneficiary-api -          │
│  /api/beneficiary/v2/payee | request/response\n        HTTP Status: 200\n    │
│  request:                                                                    │
│  {"name":"Amu","nickName":"","type":"account","accountNumber":"123456789","  │
│  ifscCode":"SBIN0000001","bankName":"STATE BANK OF                           │
│  INDIA","transactionId":"c69ef816-6227-4969-817f-b514cae15336","transaction  │
│  Type":"ADD_PAYEE"}\n          response:                                     │
│  {"status":"success","payeeId":"150813"}\n        fbf8f71c5a33c1e4 -         │
│  beneficiary-api - /admin/oauth2/introspect | request/response\n             │
│  HTTP Status: 200\n          request:                                        │
│  --4ebf8aecbd9f9d5dcc8a5fac45d956e5c78a46d18b1717530c49301d0030\r\nContent-  │
│  Disposition: form-data;                                                     │
│  name="token"\r\n\r\nory_at_N1SZMXl15zdlrS35TB6cIWQyYhX5NDyW9IaKOoUbBek.2xv  │
│  tu5kwNhYBf9yLdG5o8Qym90vlps993_Gm5N0li5I\r\n--4ebf8aecbd9f9d5dcc8a5fac45d9  │
│  56e5c78a46d18b1717530c49301d0030--\r\n\n          response:                 │
│  {"active":true,"scope":"openid                                              │
│  offline","client_id":"ui-web","sub":"d9f81da9-68f6-4394-96a3-9379368ffd9e"  │
│  ,"exp":1783678583,"iat":1783676783,"nbf":1783676783,"aud":[],"iss":"https:  │
│  //app.uat-opt.idfcfirstbank.com/platform/oauth/","token_type":"Bearer","to  │
│  ken_use":"access_token"}\n\n        9

──────────────────────────────────────────────────────────────────────────────╯

10:08:59 - LiteLLM:INFO: utils.py:3427 - 
LiteLLM completion() model= /app/models/MiniMax-M2.5; provider = openai
2026-07-12 10:08:59,166 - LiteLLM - INFO - 
LiteLLM completion() model= /app/models/MiniMax-M2.5; provider = openai
10:08:59 - LiteLLM:INFO: utils.py:3427 - 
LiteLLM completion() model= /app/models/MiniMax-M2.5; provider = openai
2026-07-12 10:08:59,168 - LiteLLM - INFO - 
LiteLLM completion() model= /app/models/MiniMax-M2.5; provider = openai
Context length exceeded. Summarizing content to fit the model context window. Might take a while...
Summarizing 2 chunks in parallel...
╭────────────────────────────── 🤖 Agent Started ──────────────────────────────╮
│                                                                              │
│  Agent: Incident Resolution Agent                                            │
│                                                                              │
│  Task: Extract customer identifiers from the incident.                       │
│                                                                              │
│  UCIC field: 1041338998                                                      │
│  Incident: Short Description: Customer is unable to do add funds             │
│  Description: Customer is unable to do add funds                             │
│                                                                              │
│  Extract these identifiers if present:                                       │
│  - ucic                                                                      │
│  - mobile_number                                                             │
│  - username                                                                  │
│  - customer_id                                                               │
│  - txn_id / txn_request_id                                                   │
│                                                                              │
│  Then use fetch_optimus_logs tool with:                                      │
│  - tag_name = the identifier type you found                                  │
│  - tag_value = the actual value                                              │
│  - service = pick from tool description based on incident type               │
│  - time_description = 'last 48 hours' unless user mentions a specific time   │
│                                                                              │
│  If no logs found with ucic, retry with mobile_number if available.          │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

10:09:04 - LiteLLM:INFO: utils.py:3427 - 
LiteLLM completion() model= /app/models/MiniMax-M2.5; provider = openai
2026-07-12 10:09:04,870 - LiteLLM - INFO - 
LiteLLM completion() model= /app/models/MiniMax-M2.5; provider = openai
10:09:11 - LiteLLM:INFO: utils.py:3427 - 
LiteLLM completion() model= /app/models/MiniMax-M2.5; provider = openai
2026-07-12 10:09:11,272 - LiteLLM - INFO - 
LiteLLM completion() model= /app/models/MiniMax-M2.5; provider = openai
10:09:11 - LiteLLM:INFO: utils.py:3427 - 
LiteLLM completion() model= /app/models/MiniMax-M2.5; provider = openai
2026-07-12 10:09:11,274 - LiteLLM - INFO - 
LiteLLM completion() model= /app/models/MiniMax-M2.5; provider = openai
10:09:11 - LiteLLM:INFO: utils.py:3427 - 
LiteLLM completion() model= /app/models/MiniMax-M2.5; provider = openai
2026-07-12 10:09:11,277 - LiteLLM - INFO - 
LiteLLM completion() model= /app/models/MiniMax-M2.5; provider = openai
Context length exceeded. Summarizing content to fit the model context window. Might take a while...
Summarizing 3 chunks in parallel...
╭────────────────────────────── 🤖 Agent Started ──────────────────────────────╮
│                                                                              │
│  Agent: Incident Resolution Agent                                            │
│                                                                              │
│  Task: Extract customer identifiers from the incident.                       │
│                                                                              │
│  UCIC field: 1041338998                                                      │
│  Incident: Short Description: Customer is unable to do add funds             │
│  Description: Customer is unable to do add funds                             │
│                                                                              │
│  Extract these identifiers if present:                                       │
│  - ucic                                                                      │
│  - mobile_number                                                             │
│  - username                                                                  │
│  - customer_id                                                               │
│  - txn_id / txn_request_id                                                   │
│                                                                              │
│  Then use fetch_optimus_logs tool with:                                      │
│  - tag_name = the identifier type you found                                  │
│  - tag_value = the actual value                                              │
│  - service = pick from tool description based on incident type               │
│  - time_description = 'last 48 hours' unless user mentions a specific time   │
│                                                                              │
│  If no logs found with ucic, retry with mobile_number if available.          │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

10:09:17 - LiteLLM:INFO: utils.py:3427 - 
LiteLLM completion() model= /app/models/MiniMax-M2.5; provider = openai
2026-07-12 10:09:17,599 - LiteLLM - INFO - 
LiteLLM completion() model= /app/models/MiniMax-M2.5; provider = openai
2026-07-12 10:09:20,138 - utils.llm - INFO - OPENAI_API_KEY refreshed
10:09:23 - LiteLLM:INFO: utils.py:3427 - 
LiteLLM completion() model= /app/models/MiniMax-M2.5; provider = openai
2026-07-12 10:09:23,539 - LiteLLM - INFO - 
LiteLLM completion() model= /app/models/MiniMax-M2.5; provider = openai
10:09:23 - LiteLLM:INFO: utils.py:3427 - 
LiteLLM completion() model= /app/models/MiniMax-M2.5; provider = openai
2026-07-12 10:09:23,540 - LiteLLM - INFO - 
LiteLLM completion() model= /app/models/MiniMax-M2.5; provider = openai
10:09:23 - LiteLLM:INFO: utils.py:3427 - 
LiteLLM completion() model= /app/models/MiniMax-M2.5; provider = openai
2026-07-12 10:09:23,546 - LiteLLM - INFO - 
LiteLLM completion() model= /app/models/MiniMax-M2.5; provider = openai
Context length exceeded. Summarizing content to fit the model context window. Might take a while...
Summarizing 3 chunks in parallel...
[CrewAIEventsBus] Warning: Event pairing mismatch. 'agent_execution_error' 
closed 'llm_call_started' (expected 'agent_execution_started')
[CrewAIEventsBus] Warning: Event pairing mismatch. 'task_failed' closed 
'agent_execution_started' (expected 'task_started')
╭────────────────────────────── 📋 Task Failure ───────────────────────────────╮
│                                                                              │
│  Task Failed                                                                 │
│  Name: Identify Customer                                                     │
│  Agent: Incident Resolution Agent                                            │
│                                                                              │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

[CrewAIEventsBus] Warning: Event pairing mismatch. 'crew_kickoff_failed' closed 
'llm_call_started' (expected 'crew_kickoff_started')


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

╭──────────────────────────────── Crew Failure ────────────────────────────────╮
│                                                                              │
│  Crew Execution Failed                                                       │
│  Name: crew                                                                  │
│  ID: 38634691-e978-4971-9537-8277b85838de                                    │
│                                                                              │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

[CrewAIEventsBus] Warning: Event pairing mismatch. 'method_execution_failed' 
closed 'agent_execution_started' (expected 'method_execution_started')
2026-07-12 10:09:29,520 - crewai.flow.flow - ERROR - Error executing listener run_resolver_crew: LLM context length exceeded. Original error: litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError: OpenAIException - litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError: Hosted_vllmException - This model's maximum context length is 150000 tokens. However, you requested 0 output tokens and your prompt contains at least 150001 input tokens, for a total of at least 150001 tokens. Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_tokens, value=150001)
model=/app/models/MiniMax-M2.5. context_window_fallbacks=None. fallbacks=None.

Set 'context_window_fallback' - https://docs.litellm.ai/docs/routing#fallbacks. Received Model Group=/app/models/MiniMax-M2.5
Available Model Group Fallbacks=None
Consider using a smaller input or implementing a text splitting strategy.
╭─────────────────────────── ❌ Flow Method Failed ────────────────────────────╮
│                                                                              │
│  Method: run_resolver_crew                                                   │
│  Status: Failed                                                              │
│                                                                              │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

2026-07-12 10:09:29,523 - __main__ - ERROR - ✗ Flow FAILED | incident=4470398 error=LLM context length exceeded. Original error: litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError: OpenAIException - litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError: Hosted_vllmException - This model's maximum context length is 150000 tokens. However, you requested 0 output tokens and your prompt contains at least 150001 input tokens, for a total of at least 150001 tokens. Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_tokens, value=150001)
model=/app/models/MiniMax-M2.5. context_window_fallbacks=None. fallbacks=None.

Set 'context_window_fallback' - https://docs.litellm.ai/docs/routing#fallbacks. Received Model Group=/app/models/MiniMax-M2.5
Available Model Group Fallbacks=None
Consider using a smaller input or implementing a text splitting strategy.
2026-07-12 10:09:30,658 - __main__ - INFO -   Fallback rejection sent | incident=4470398
shishir.pandey_tho@0325LTPB0124444 ~ 
