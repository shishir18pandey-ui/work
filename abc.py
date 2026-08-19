2026-08-19 07:13:31,453 - __main__ - INFO - Timeout checker thread started
2026-08-19 07:13:31,454 - __main__ - INFO - ✓ Worker started
2026-08-19 07:13:31,454 - __main__ - INFO -   Topic    : GEN-AI-DE-INCIDENT-EVENTS
2026-08-19 07:13:31,454 - __main__ - INFO -   Group    : gen-ai-de-incident-managers
2026-08-19 07:13:31,454 - __main__ - INFO -   Broker   : b-1.dcawscentraluatkafka0.fil3x5.c4.kafka.ap-south-1.amazonaws.com:9096,b-2.dcawscentraluatkafka0.fil3x5.c4.kafka.ap-south-1.amazonaws.com:9096,b-3.dcawscentraluatkafka0.fil3x5.c4.kafka.ap-south-1.amazonaws.com:9096
2026-08-19 07:13:31,548 - new_flow.utils.llm - INFO - OPENAI_API_KEY refreshed
2026-08-19 07:23:31,628 - new_flow.utils.llm - INFO - OPENAI_API_KEY refreshed
2026-08-19 07:33:31,681 - new_flow.utils.llm - INFO - OPENAI_API_KEY refreshed
2026-08-19 07:37:05,344 - __main__ - INFO - → Message received | incident=0013632 event=new_incident partition=0 offset=496
2026-08-19 07:37:05,344 - __main__ - INFO - ✓ Offset committed | incident=0013632
2026-08-19 07:37:05,365 - __main__ - INFO - → Processing | incident=0013632 event=new_incident module=new_flow.flow
2026-08-19 07:37:05,490 - __main__ - INFO -   DB status | incident=0013632 status=in_progress
2026-08-19 07:37:07,073 - crewai.cli.config - INFO - Using config path: /root/.config/crewai/settings.json
2026-08-19 07:37:07,496 - new_flow.tools.service_metadata - INFO - Loaded service metadata for apps: ['optimus', 'cbs', 'idp']
2026-08-19 07:37:09,389 - __main__ - INFO -   Flow type | incident=0013632 type=new_incident
ot-collector.tracing.svc.cluster.local tetsing1
╭───────────────────────────── 🌊 Flow Execution ──────────────────────────────╮
│                                                                              │
│  Starting Flow Execution                                                     │
│  Name: IncidentManagementFlow                                                │
│  ID: fb2f5b3a-cb03-4b51-90be-5295b4cd7abc                                    │
│                                                                              │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

╭────────────────────────────── 🌊 Flow Started ───────────────────────────────╮
│                                                                              │
│  Flow Started                                                                │
│  Name: IncidentManagementFlow                                                │
│  ID: fb2f5b3a-cb03-4b51-90be-5295b4cd7abc                                    │
│                                                                              │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

Flow started with ID: fb2f5b3a-cb03-4b51-90be-5295b4cd7abc
2026-08-19 07:37:25,425 - crewai.flow.flow - INFO - Flow started with ID: fb2f5b3a-cb03-4b51-90be-5295b4cd7abc
╭─────────────────────────── 🔄 Flow Method Running ───────────────────────────╮
│                                                                              │
│  Method: initialize_and_classify                                             │
│  Status: Running                                                             │
│                                                                              │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

2026-08-19 07:37:25,431 - new_flow.agents.intent_classifier - INFO - [TOKEN CHECK] using token ending in kxAS9ss4
Generated incident description for UCIC 
=== Intent Classifier LLM ===
Model: openai//app/models/Qwen3-14B-FP8
Base URL: https://llm-api.iservebetter.idfcfirstbank.com/qwen3-14b-entauth/v1
╭───────────────────────── 🚀 Crew Execution Started ──────────────────────────╮
│                                                                              │
│  Crew Execution Started                                                      │
│  Name: crew                                                                  │
│  ID: ede9b650-52e1-460e-a255-27e523a6542a                                    │
│                                                                              │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

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
│  verification                                                                │
│  Description: Customer is unable to do verificstion                          │
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
│  ID: ea7c0886-ee2c-4407-a628-8877cf77688d                                    │
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
│  verification                                                                │
│  Description: Customer is unable to do verificstion                          │
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


07:37:41 - LiteLLM:INFO: utils.py:3427 - 
LiteLLM completion() model= /app/models/Qwen3-14B-FP8; provider = openai
2026-08-19 07:37:41,494 - LiteLLM - INFO - 
LiteLLM completion() model= /app/models/Qwen3-14B-FP8; provider = openai
╭─────────────────────────── ✅ Agent Final Answer ────────────────────────────╮
│                                                                              │
│  Agent: Intent Classifier                                                    │
│                                                                              │
│  Final Answer:                                                               │
│  {                                                                           │
│    "intent": "additional_info",                                              │
│    "user_goal": "To report an issue with verification functionality on the   │
│  platform.",                                                                 │
│    "issue_description": "The customer is unable to complete verification     │
│  due to an issue involving the execution of a script, possibly related to    │
│  cross-site scripting (XSS).",                                               │
│    "problem_summary": "The user has reported that a customer is unable to    │
│  perform verification on the platform. The issue is described with a script  │
│  tag containing an alert for XSS, which may be causing the verification      │
│  process to fail. The description also mentions that the customer is unable  │
│  to complete verification, indicating a potential technical issue with the   │
│  verification system or its interaction with scripts."                       │
│  }                                                                           │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

╭───────────────────────────── 📋 Task Completion ─────────────────────────────╮
│                                                                              │
│  Task Completed                                                              │
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
│  verification                                                                │
│  Description: Customer is unable to do verificstion                          │
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

╭────────────────────────────── Crew Completion ───────────────────────────────╮
│                                                                              │
│  Crew Execution Completed                                                    │
│  Name: crew                                                                  │
│  ID: ede9b650-52e1-460e-a255-27e523a6542a                                    │
│  Final Output: {                                                             │
│    "intent": "additional_info",                                              │
│    "user_goal": "To report an issue with verification functionality on the   │
│  platform.",                                                                 │
│    "issue_description": "The customer is unable to complete verification     │
│  due to an issue involving the execution of a script, possibly related to    │
│  cross-site scripting (XSS).",                                               │
│    "problem_summary": "The user has reported that a customer is unable to    │
│  perform verification on the platform. The issue is described with a script  │
│  tag containing an alert for XSS, which may be causing the verification      │
│  process to fail. The description also mentions that the customer is unable  │
│  to complete verification, indicating a potential technical issue with the   │
│  verification system or its interaction with scripts."                       │
│  }                                                                           │
│                                                                              │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

2026-08-19 07:37:42,878 - opentelemetry.attributes - WARNING - Invalid type NoneType for attribute 'sop_value' value. Expected one of ['bool', 'str', 'bytes', 'int', 'float'] or a sequence of those types
Enhanced classifier | incident=0013632 | app= | category=kyc_issue
Initialized and classified incident 0013632 with intent: additional_info
Need more info for incident 0013632
╭────────────────────────── ✅ Flow Method Completed ──────────────────────────╮
│                                                                              │
│  Method: initialize_and_classify                                             │
│  Status: Completed                                                           │
│                                                                              │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

╭─────────────────────────── 🔄 Flow Method Running ───────────────────────────╮
│                                                                              │
│  Method: start_process                                                       │
│  Status: Running                                                             │
│                                                                              │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

╭────────────────────────── ✅ Flow Method Completed ──────────────────────────╮
│                                                                              │
│  Method: start_process                                                       │
│  Status: Completed                                                           │
│                                                                              │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

╭───────────────────────────── ✅ Flow Completion ─────────────────────────────╮
│                                                                              │
│  Flow Execution Completed                                                    │
│  Name: IncidentManagementFlow                                                │
│  ID: fb2f5b3a-cb03-4b51-90be-5295b4cd7abc                                    │
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

2026-08-19 07:37:42,887 - __main__ - INFO - ✓ Flow completed | incident=0013632
shishir.pandey_tho@0325LTPB0124444 ~ % 



 "assignmentGroup": "",
    "businessCorrectiveAction": "",
    "businessImpact": " Customer clicked http://apple-site.com?redirect=\u003cimg src=x onerror=alert(1)\u003e \u0026 lost access \"urgently\" \u003cscript\u003ealert(\u0027XSS\u0027)\u003c/script\u003e Customer unable to login",
    "businessPreventiveAction": "",
    "businessService": "SFDC Asset Org 3",
    "callerId": "shyam.kc@idfcfirstbank.com",
    "cause": "",
    "contactType": "Self Service",
    "created_at": "2026-08-19T07:37:05.231143",
    "dataSource": "",
    "description": "Customer is unable to do verificstion",
    "descriptionOfOutage": "",
    "entityUCIC": "",
    "file_description": "[Attached image: pan_card.jpeg]\nThere was a problem with your transaction\nPlease try adding funds with a different method.\nRetry\nSee more deposit methods",
    "hashValues": "",
    "headers": {
        "Content-Type": "application/json",
        "correlationId": "13ceda18-d0cd-4f9b-b249-c1f5fe2ed0d6",
        "source": "IncidentBot",
        "transactionId": "3e330aee-e2e0-4e8f-8af9-0eb887a3c98c"
    },
    "impact": "Low",
    "incidentId": "0013632",
    "incidentNumber": "INC000006212701",
    "incidentType": "Application",
    "individualUCIC": "",
    "interaction_counter": 0,
    "ipDetails": "",
    "ldwNotifyInformation": "",
    "loanAccountNumber": "53637373733838",
    "loginId": "",
    "mobileNumber": "",
    "resoultionTeam": "",
    "rootCause": "",
    "shortDescription": "\u003cscript\u003ealert(\u0027XSS\u0027)\u003c/script\u003e Customer is unable to do verification ",
    "sourceIncCreateddttime": "24-Feb-2023 09:10:04",
    "sourceIncidentId": "",
    "sourceIncidentNum": "INC000006212999",
    "state": "New",
    "status": "created",
    "systemName": "",
    "techCorrectiveAction": "",
    "techPreventiveAction": "",
    "tier1": "Rural - Home Loan",
    "tier2": "Data Verification Issue",
    "tier3": "Something Went Wrong , ALOG Error",
    "urgency": "Medium",
    "urlOrDomain": "",
    "userDetail": "",
    "userLocation": ""
}


                                                         
