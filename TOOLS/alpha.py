

Now I have a comprehensive understanding of the entire application. Let me provide you with a complete flow explanation:

---

# 🔄 Complete Incident Manager Application Flow

## 1. 📥 **Entry Point: Kafka Message Processing**

```
┌─────────────────────────────────────────────────────────────────────┐
│                     WORKER.PY (Kafka Consumer)                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Kafka Topic: GEN-AI-DE-INCIDENT-EVENTS                              │
│                                                                     │
│  Message Format:                                                    │
│  {                                                                   │
│    "incident_id": "INC001",                                         │
│    "event_type": "new_incident",                                    │
│    "payload": { ... incident data from ServiceNow ... }            │
│  }                                                                   │
│                                                                     │
│  Flow Selection (via USE_FLOW env var):                              │
│  ├── "old"        → flow.py (deprecated)                           │
│  ├── "new"        → new_flow/flow.py (Jaeger-first)                │
│  └── "elk_search" → elk_search_flow/flow.py (ELK-first span search)│
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
```

---

## 2. 🎯 **Stage 1: Intent Classification & Enrichment**

```
┌─────────────────────────────────────────────────────────────────────┐
│               INTENT CLASSIFIER (new_flow/agents/)                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input:                                                              │
│  - incident_description (from payload)                              │
│  - user_qa_pairs (previous Q&A from customer)                      │
│  - comment (latest customer interaction)                           │
│  - previous_classifier_output (if re-processing)                  │
│                                                                     │
│  Processing:                                                         │
│  1. extract_identifiers() - Regex-based extraction                 │
│     └── UCIC, mobile, account, loan_account, customer_id            │
│                                                                     │
│  2. guess_problem_category() - Pattern matching                   │
│     └── account_freeze, login_failure, transaction_failed, etc.    │
│                                                                     │
│  3. run_intent_classifier_crew_async() - LLM Agent                │
│     └── Determines:                                                 │
│         • intent (closure/rebuttal/additional_info)                 │
│         • app (cbs/optimus/idp/collections)                         │
│         • problem_category                                          │
│         • needs_user_input                                          │
│         • clarification_question                                    │
│         • user_goal, issue_description, problem_summary             │
│                                                                     │
│  Output: IntentClassifierOutput                                    │
│  {                                                                   │
│    "intent": "additional_info",                                     │
│    "app": "optimus",                                                │
│    "problem_category": "login_failure",                            │
│    "customer_identifiers": {"ucic": "...", "mobile": "..."},       │
│    "enriched_prompt": "...",                                        │
│    "needs_user_input": false                                        │
│  }                                                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
```

---

## 3. 🔍 **Stage 2: RAG - Historic Context (Similarity Search)**

```
┌─────────────────────────────────────────────────────────────────────┐
│                 CONTEXT BUILDER (Semantic Search RAG)                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  This is the RAG pipeline for finding similar historic incidents:  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  SEMANTIC SEARCH (via SEMANTIC_SEARCH_ENDPOINT)            │   │
│  │                                                             │   │
│  │  Endpoint: os.getenv("SEMANTIC_SEARCH_ENDPOINT")           │   │
│  │                                                             │   │
│  │  Request:                                                  │   │
│  │  {                                                          │   │
│  │    "metadata": {"application": "optimus"},                │   │
│  │    "query": "<incident_description>",                      │   │
│  │    "index_source": "incidents"                              │   │
│  │  }                                                          │   │
│  │                                                             │   │
│  │  Response:                                                  │   │
│  │  [                                                          │   │
│  │    {"document": {"id": "INC123", "content": "..."},         │   │
│  │     "score": 0.85,  ← SIMILARITY SCORE (0-1)                │   │
│  │     "chunked_content": "..."},                              │   │
│  │    ...                                                       │   │
│  │  ]                                                          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  format_incidents_for_llm()                                  │   │
│  │                                                             │   │
│  │  Formats with similarity score embedded:                   │   │
│  │  "=== SIMILAR INCIDENT 1 (Similarity: 85.00%) ==="          │   │
│  │  Incident ID: INC123                                        │   │
│  │  RESOLUTION: [resolution notes]                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  LLM Crew Analysis (HistoricContext)                         │   │
│  │  - Summarizes similar incidents                             │   │
│  │  - Extracts relevant resolutions                            │   │
│  │  - Returns top_similarity_pct (from raw results)            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Output: HistoricContext                                            │
│  {                                                                   │
│    "text": "Summary of similar incidents...",                       │
│    "top_similarity_pct": 85.0                                       │
│  }                                                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
```

---

## 4. 🛠️ **Stage 3: Service Discovery**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DISCOVERY TOOLS                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  get_services_impl(app)                                      │  │
│  │  - Returns predefined services from app_config.json         │  │
│  │  - optimus, cbs, idp, collections, etc.                     │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  discover_jaeger_services_impl(app)                         │  │
│  │  - Dynamic discovery from Jaeger API                       │  │
│  │  - GET /api/services                                        │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                              │                                      │              │
│                              ▼                                      │              │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  discover_jaeger_tags_impl(service, app)                   │  │
│  │  - Dynamic tag discovery from Jaeger                       │  │
│  │  - GET /api/tags?service={service}                         │  │
│  │  - Returns: customer_id, txn_id, mobile_number, etc.        │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
```

---

## 5. 📋 **Stage 4: Plan Agent**

```
┌─────────────────────────────────────────────────────────────────────┐
│                      PLAN AGENTS                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input:                                                              │
│  - incident_context (historic similarity search)                    │
│  - customer_identifiers                                             │
│  - problem_category                                                 │
│  - app                                                              │
│  - discovered_services                                              │
│                                                                     │
│  Processing:                                                         │
│  1. Analyzes historic context for patterns                          │
│  2. Identifies suggested Jaeger service to query                   │
│  3. Determines if more information is needed                       │
│  4. Validates service names against discovered list                │
│                                                                     │
│  Output: PlanOutput                                                  │
│  {                                                                   │
│    "issue_identified": true,                                        │
│    "issue_summary": "Login failure due to...",                      │
│    "next_steps": "Query idp-api service with customer_id",           │
│    "needs_more_info": false,                                         │
│    "suggested_service": "idp-api",                                   │
│    "suggested_services": ["idp-api", "mfa-api", "upi-api"],        │
│    "customer_identifiers": {"customer_id": "..."}                  │
│  }                                                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
```

---

## 6. 🔎 **Stage 5: Execution - Deep Trace Investigation**

### **A. Jaeger-First Approach (new_flow)**

```
┌─────────────────────────────────────────────────────────────────────┐
│              EXECUTE AGENT - JAEGER (new_flow)                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  AUTO-LOOP with TIME RANGE ITERATION:                               │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  MAX 7 TIME RANGES (24h each)                               │   │
│  │                                                             │   │
│  │  time_range_index:                                           │   │
│  │  ├── 0 = 0-24h ago (most recent)                            │   │
│  │  ├── 1 = 24-48h ago                                          │   │
│  │  ├── 2 = 48-72h ago                                          │   │
│  │  ├── 3 = 72-96h ago                                          │   │
│  │  ├── 4 = 96-120h ago                                         │   │
│  │  ├── 5 = 120-144h ago                                        │   │
│  │  └── 6 = 144-168h ago (oldest)                               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  TOOL: JaegerTraceTool                                      │   │
│  │                                                             │   │
│  │  Query Parameters:                                          │   │
│  │  - service: "idp-api"                                       │   │
│  │  - tag_name: "customer_id" (auto-tries ucic too)           │   │
│  │  - tag_value: "<customer_id>"                              │   │
│  │  - time_range_index: 0-6                                     │   │
│  │                                                             │   │
│  │  Error Detection:                                            │   │
│  │  - http.status_code != 2xx                                   │   │
│  │  - error tag = true                                          │   │
│  │  - error-level log events                                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  AUTO-ADVANCE LOGIC:                                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  if no traces found:                                        │   │
│  │      current_time_range_index += 1  (auto-advance)         │   │
│  │      → query next older time window                        │   │
│  │                                                             │   │
│  │  if traces found:                                           │   │
│  │      → analyze error hierarchy                             │   │
│  │      → identify root cause                                  │   │
│  │      → proceed to resolution                                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  OUTPUT: JaegerExecutionResult                                       │
│  {                                                                   │
│    "resolved": true/false,                                          │
│    "diagnosis": "Root cause analysis",                              │
│    "solution": "Resolution steps",                                   │
│    "questions": ["clarification questions"],                      │
│    "tool_calls": [{"tool_name": "...", "output": "..."}],           │
│    "confidence": 0.85,                                              │
│    "iterations_completed": 3,                                       │
│    "ranges_exhausted_per_iteration": [1, 2, 3]                      │
│  }                                                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### **B. ELK-First Approach (elk_search_flow)**

```
┌─────────────────────────────────────────────────────────────────────┐
│                 EXECUTE - ELK-FIRST (elk_search_flow)               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Instead of scanning 7 time windows, uses keyword search in        │
│  Elasticsearch span index to locate candidate traces:             │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  SPAN KEYWORD SEARCH                                        │   │
│  │  ┌─────────────────────────────────────────────────────┐  │   │
│  │  │ elk_search_flow/tools/span_search.py                 │  │   │
│  │  │                                                         │  │   │
│  │  │ 1. Build keyword query from incident description      │  │   │
│  │  │ 2. Search span index for matching spans               │  │   │
│  │  │ 3. Group by trace_id                                   │  │   │
│  │  │ 4. Rank by error presence + relevance                 │  │   │
│  │  └─────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  FETCH SPAN TREE                                           │   │
│  │  - Get full trace for top candidates                       │   │
│  │  - Build error chain hierarchy                            │   │
│  │  - Extract diagnostics                                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  FRONTEND TRACES (optional)                                 │   │
│  │  - Search mobile/web frontend traces                       │   │
│  │  - Find user session, device info                          │   │
│  │  - Cross-reference with backend spans                      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 7. ⚙️ **Tools Available for Investigation**

```
┌─────────────────────────────────────────────────────────────────────┐
│                        QUERY TOOLS                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  1. JaegerTraceTool                                        │   │
│  │     - Search traces by tag (customer_id, ucic, txn_id)     │   │
│  │     - Auto-tag fallback (ucic → customer_id)               │   │
│  │     - Time range iteration (7 windows of 24h)               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  2. ELKQueryTool                                           │   │
│  │     - Query application logs in Elasticsearch             │   │
│  │     - Index patterns: elk-{app}-prod*, etc.               │   │
│  │     - Fields: customer_id, txn_id, message, logtype        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  3. SQLQueryTool (Oracle)                                  │   │
│  │     - Execute SELECT queries on CBS/IDP databases          │   │
│  │     - VALIDATION: query_validator.py                       │   │
│  │       ├── Blocked: DROP, TRUNCATE, GRANT, REVOKE, ALTER    │   │
│  │       └── Requires WHERE for DELETE/UPDATE                 │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  4. GetTableSchemaTool                                     │   │
│  │     - Get column names and types                           │   │
│  │     - Query: all_tab_columns                               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  5. SimilaritySearchTool                                    │   │
│  │     - Search similar historic incidents                   │   │
│  │     - Uses SEMANTIC_SEARCH_ENDPOINT                        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  6. Discovery Tools                                        │   │
│  │     - discover_jaeger_services                             │   │
│  │     - discover_jaeger_tags                                │   │
│  │     - discover_elk_fields                                  │   │
│  │     - get_app_config                                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 8. ✅ **Stage 6: Self-Critique & Confidence**

```
┌─────────────────────────────────────────────────────────────────────┐
│                      SELF-CRITIQUE AGENT                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input:                                                              │
│  - incident_description                                             │
│  - diagnosis (from execution)                                       │
│  - solution (from execution)                                        │
│  - tool_calls (all queries made)                                    │
│  - user_qa_pairs                                                    │
│                                                                     │
│  Processing:                                                         │
│  Evaluates:                                                          │
│  1. Does diagnosis match incident?                                  │
│  2. Is solution actionable and complete?                            │
│  3. Any gaps or inconsistencies?                                    │
│  4. Could solution cause harm?                                     │
│                                                                     │
│  Output: SelfCritiqueOutput                                         │
│  {                                                                   │
│    "is_valid": true/false,                                          │
│    "confidence": 0.0-1.0,           ← CONFIDENCE SCORE              │
│    "issues_found": ["gap1", "gap2"],                               │
│    "suggested_fixes": ["fix1", "fix2"],                             │
│    "revised_diagnosis": "...",                                      │
│    "revised_solution": "..."                                        │
│  }                                                                   │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  ESCALATION DECISION: should_escalate()                   │   │
│  │                                                             │   │
│  │  Escalate if:                                               │   │
│  │  ├── is_valid = false                                      │   │
│  │  ├── confidence < 0.4                                      │   │
│  │  └── issues_found >= 3                                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 9. 📝 **Stage 7: Summary Agent**

```
┌─────────────────────────────────────────────────────────────────────┐
│                       SUMMARY AGENT                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Uses confidence from multiple sources:                             │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  CONFIDENCE CALCULATION:                                    │   │
│  │                                                             │   │
│  │  1. top_similarity_pct (from RAG search, raw score)        │   │
│  │     └── Carried from HistoricContext.top_similarity_pct    │   │
│  │                                                             │   │
│  │  2. Jaeger trace findings (execution agent)                │   │
│  │     └── 100% if traces found + error identified           │   │
│  │                                                             │   │
│  │  3. Self-critique confidence                               │   │
│  │     └── 0-1 score from QA agent                           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  OUTPUT: SummaryOutput                                     │   │
│  │                                                             │   │
│  │  {                                                          │   │
│  │    "diagnosis": "Root cause of the issue",                │   │
│  │    "solution": "Steps to resolve",                         │   │
│  │    "questions": ["Clarification question if needed"],      │   │
│  │    "resolved": "yes/no"                                    │   │
│  │  }                                                          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 10. 🔄 **Auto-Loop & Error Handling Strategy**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AUTO-LOOP MECHANISMS                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1️⃣  TIME RANGE AUTO-ADVANCE (Jaeger):                              │
│      ┌─────────────────────────────────────────────────────────┐   │
│      │  Loop while:                                             │   │
│      │    - no traces found in current window                  │   │
│      │    - current_time_range_index < 6                       │   │
│      │                                                             │   │
│      │  Action:                                                  │   │
│      │    current_time_range_index += 1                         │   │
│      │    → query next 24h window                               │   │
│      └─────────────────────────────────────────────────────────┘   │
│                                                                     │
│  2️⃣  TAG AUTO-FALLBACK (Jaeger):                                    │
│      ┌─────────────────────────────────────────────────────────┐   │
│      │  If tag_name fails:                                      │   │
│      │    - Try: ucic → customer_id → individualucic → ...      │   │
│      │    - Returns results from first successful tag          │   │
│      └─────────────────────────────────────────────────────────┘   │
│                                                                     │
│  3️⃣  SERVICE AUTO-ITERATION:                                         │
│      ┌─────────────────────────────────────────────────────────┐   │
│      │  If current service yields nothing:                    │   │
│      │    - Add next service from suggested_services          │   │
│      │    - Always include: idp-api, mfa-api                  │   │
│      └─────────────────────────────────────────────────────────┘   │
│                                                                     │
│  4️⃣  LLM TOKEN REFRESH (on error):                                  │
│      ┌─────────────────────────────────────────────────────────┐   │
│      │  If API call fails:                                      │   │
│      │    - Refresh auth token via ENT_AUTH                   │   │
│      │    - Retry once                                         │   │
│      └─────────────────────────────────────────────────────────┘   │
│                                                                     │
│  5️⃣  CONTEXT COMPRESSION (for long runs):                           │
│      ┌─────────────────────────────────────────────────────────┐   │
│      │  If token limit exceeded:                               │   │
│      │    - Keep last 3 tool calls fully                       │   │
│      │    - Summarize older calls to key info                 │   │
│      │    - Cap at 200k tokens (50k chars)                    │   │
│      └─────────────────────────────────────────────────────────┘   │
│                                                                     │
│  6️⃣  CIRCUIT BREAKER (tool failures):                               │
│      ┌─────────────────────────────────────────────────────────┐   │
│      │  If tool fails N times in a row:                        │   │
│      │    - Mark service as temporarily unavailable           │   │
│      │    - Skip and try next tool                             │   │
│      │    - Reset after timeout                                │   │
│      └─────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 11. 📤 **Stage 8: ServiceNow Integration**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SERVICENOW UPDATE                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Based on final intent:                                             │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  INTENT: "closure" → RESOLVED                              │   │
│  │  - diagnosis & solution sent to customer                  │   │
│  │  - incident marked as resolved                            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  INTENT: "additional_info" → QUESTION                       │   │
│  │  - Clarification question sent to customer                  │   │
│  │  - Waits for response                                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  INTENT: "rebuttal" / ESCALATION → REJECT                   │   │
│  │  - Assign to Engineer                                       │   │
│  │  - Payload updated with logs                                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Payload stored in DB:                                             │
│  - upsert_incident_payload_async(incident_id, payload_json)        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 12. 🏗️ **Complete Flow Diagram**

```
                    ┌──────────────────────────────┐
                    │    KAFKA MESSAGE RECEIVED    │
                    │   (incident_id, payload)     │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │   INTENT CLASSIFIER           │
                    │  • Extract identifiers       │
                    │  • Guess problem category   │
                    │  • Determine intent         │
                    └──────────────┬───────────────┘
                                   │
          ┌────────────────────────┴────────────────────────┐
          │                                                   │
┌─────────▼─────────────┐                          ┌─────────▼─────────────┐
│  RAG (Semantic Search)│                          │  DISCOVERY TOOLS     │
│  • Similar incidents  │                          │  • Get services      │
│  • Similarity scores  │                          │  • Discover tags     │
│  • top_similarity_pct │                          └──────────┬────────────┘
└──────────┬────────────┘                                   │
           │                                                │
           └────────────┬───────────────────────────────────┘
                        │
           ┌────────────▼─────────────┐
           │    PLAN AGENT            │
           │  • Issue identification  │
           │  • Service selection     │
           │  • Next steps            │
           └────────────┬─────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
┌───────▼────────┐            ┌────────▼─────────┐
│   NEW FLOW     │            │  ELK_SEARCH_FLOW │
│   (Jaeger)    │            │    (ELK-first)   │
└───────┬────────┘            └────────┬─────────┘
        │                             │
        │  ┌──────────────────────┐   │   ┌───────────────────────┐
        │  │ AUTO-LOOP:           │   │   │ SPAN SEARCH:          │
        │  │ • 7 time ranges      │   │   │ • Keyword search      │
        │  │ • Tag fallback      │   │   │ • Trace ranking       │
        │  │ • Auto-advance      │   │   │ • Fetch span tree     │
        │  └──────────────────────┘   │   └───────────────────────┘
        │                             │
        └─────────────┬───────────────┘
                      │
           ┌──────────▼──────────┐
           │  EXECUTION AGENT   │
           │  • Diagnosis       │
           │  • Solution        │
           │  • Tool calls      │
           └─────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │  SELF-CRITIQUE     │
          │  • is_valid        │
          │  • confidence      │
          │  • issues_found    │
          └─────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │  SUMMARY AGENT     │
          │  • Final output   │
          │  • Conf. score    │
          └─────────┬──────────┘
                     │
          ┌─────────▼──────────┐
          │  SERVICENOW UPDATE │
          │  • closure        │
          │  • question       │
          │  • rejection      │
          └───────────────────┘
```

---

## 📊 **Key Components Summary**

| Component | Purpose | Files |
|-----------|----------|-------|
| **Worker** | Kafka consumer, flow selection | `worker.py` |
| **Intent Classifier** | Extract intent, app, identifiers | `agents/intent_classifier.py` |
| **RAG/Context Builder** | Historic incident search via semantic search | `agents/context_builder.py` |
| **Discovery Tools** | Find services & tags | `tools/discovery_tools.py` |
| **Plan Agent** | Plan investigation strategy | `agents/plan_agents.py` |
| **Execute Agent** | Deep trace investigation with auto-loop | `agents/execute_agent_jaeger.py` |
| **Self-Critique** | Validate diagnosis/solution | `agents/self_critique.py` |
| **Summary Agent** | Generate final output with confidence | `agents/summary_agent.py` |
| **Query Tools** | Jaeger, ELK, SQL, Similarity search | `tools/query_tools.py` |
| **Query Validator** | SQL safety validation | `tools/query_validator.py` |
| **LLM Utils** | Token management, retry, caching | `utils/llm.py`, `llm_cache.py` |
| **Context Compression** | Token budget management | `utils/context_compression.py` |

This is a sophisticated **Agentic AI system** with multiple auto-loops, confidence scoring from multiple sources, RAG-based historic context, and comprehensive error handling!
































Now let me analyze how this project validates against each of these lessons:

---

# 📚 AI Engineering Lessons Validation

## Lesson 1: Maximum Useful Leverage ✅ VALIDATED

**Principle**: Maximize AI's capabilities - use it for complex reasoning, not just simple tasks.

### How This Project Does It:

| Aspect | Implementation | Validation |
|--------|---------------|------------|
| **LLM for Reasoning** | Intent classification, self-critique, summary generation | ✅ Full AI reasoning |
| **Multiple Agents** | Classifier, Plan, Execute, Summary, Self-Critique agents | ✅ Specialized roles |
| **Tool-augmented AI** | Jaeger, ELK, SQL, Semantic Search tools | ✅ AI + external data |
| **Context Enrichment** | Customer identifiers, historic incidents, discovered services | ✅ Rich context |

```python
# Example: Intent classifier uses LLM for nuanced decision making
intent_result = await run_intent_classifier_crew_async(
    incident_description,
    user_qa_pairs,
    interaction,
    previous_classifier_output=previous_classifier_output,
)
```

**Status**: ✅ **FULLY VALIDATED** - Project uses AI for complex reasoning, not just simple automation.

---

## Lesson 2: Craftsmanship ✅ VALIDATED

**Principle**: Use AI carefully - quality over quantity, careful prompting, thorough testing.

### How This Project Does It:

| Aspect | Implementation | Validation |
|--------|---------------|------------|
| **Detailed Prompts** | Role-specific prompts in each agent | ✅ Rich backstories |
| **Output Validation** | Self-critique agent validates diagnosis | ✅ Quality checks |
| **Confidence Scoring** | Multiple sources of confidence | ✅ Measured approach |
| **Pydantic Models** | Structured outputs with type validation | ✅ Contract testing |

```python
# Example: Structured output with validation
class IntentClassifierOutput(BaseModel):
    intent: str
    app: str
    problem_category: str
    customer_identifiers: Dict[str, str]
    enriched_prompt: str
    confidence: float  # Explicit confidence
```

**Status**: ✅ **FULLY VALIDATED** - Careful agent design with validation.

---

## Lesson 3: Don't Ship Bad AI-Generated Software ⚠️ PARTIALLY VALIDATED

**Principle**: Don't ship AI-generated code without human review - can have subtle bugs.

### How This Project Addresses It:

| Aspect | Implementation | Status |
|--------|---------------|--------|
| **Query Validator** | Blocks DROP, TRUNCATE, GRANT, ALTER | ✅ Safety |
| **Type Validation** | Pydantic models for all inputs | ✅ Type safety |
| **Error Handling** | Try-except blocks, fallback responses | ✅ Defensive |
| **Testing** | Test files exist but coverage unclear | ⚠️ Unknown |

```python
# Example: SQL Query Safety Validation
BLOCKED_KEYWORDS = ["DROP", "TRUNCATE", "GRANT", "REVOKE", "ALTER"]

def validate_sql_query(query: str) -> Dict[str, Any]:
    # Block dangerous operations
    for keyword in BLOCKED_KEYWORDS:
        if keyword in query_normalized:
            return {"is_safe": False, "reason": f"Blocked: {keyword}"}
```

**Status**: ⚠️ **PARTIALLY VALIDATED** - Safety mechanisms exist, but need better:
- Unit test coverage
- Integration testing
- A/B testing for prompts
- Human review gates

---

## Lesson 4: Source of Truth & Autoloops ✅ FULLY VALIDATED

**Principle**: AI must have reliable source of truth; use autoloops for self-correction.

### How This Project Does It:

| Aspect | Implementation | Validation |
|--------|---------------|------------|
| **Source of Truth** | Real Jaeger traces, ELK logs, Oracle DB | ✅ Live systems |
| **Autoloop 1: Time Range** | Auto-advance 7 time windows (24h each) | ✅ Auto-retry |
| **Autoloop 2: Tag Fallback** | ucic → customer_id → individualucic | ✅ Auto-fallback |
| **Autoloop 3: Service Iteration** | Try next service if current fails | ✅ Auto-switch |
| **Autoloop 4: LLM Retry** | Refresh token + retry on API failure | ✅ Auto-retry |
| **Autoloop 5: Context Compression** | Summarize old tool calls | ✅ Auto-manage |

```python
# Autoloop 1: Time Range Auto-Advance
if self._is_no_traces_result(result):
    if self.current_time_range_index < 6:
        self.current_time_range_index += 1  # AUTO-ADVANCE
        logger.info(f"No traces at range {range_index}, advancing to {self.current_time_range_index}")

# Autoloop 2: Tag Fallback
attempt_order = [tag_name]
if tag_name in ("ucic", "customer_id", "individualucic", "entityucic"):
    for candidate in ("ucic", "customer_id"):
        if candidate not in attempt_order:
            attempt_order.append(candidate)  # AUTO-FALLBACK
```

**Status**: ✅ **FULLY VALIDATED** - Multiple autoloops for self-correction with real source of truth (Jaeger/ELK/DB).

---

## Lesson 5: Not Directly Present in Project

This lesson is about something else - doesn't apply to this project analysis.

---

## Lesson 6: Long-Running Agent ✅ VALIDATED

**Principle**: AI agents must handle long-running operations gracefully.

### How This Project Does It:

| Aspect | Implementation | Validation |
|--------|---------------|------------|
| **Timeout Handling** | TIMEOUT_SECONDS = 14400 (4 hours) | ✅ Long timeout |
| **Token Management** | Context compression for long runs | ✅ Budget management |
| **State Persistence** | State stored in DB between steps | ✅ Checkpointing |
| **Async Processing** | asyncio for concurrent operations | ✅ Non-blocking |
| **Error Recovery** | Rebuttal handling, requeue logic | ✅ Resilience |

```python
# Long-running timeout
TIMEOUT_SECONDS = 14400  # 4 hours for complex queries

# Context compression for long runs
def compress_context(context, tool_calls, max_tokens=200000):
    # Summarizes older tool calls
    # Keeps recent calls fully
    # Prevents token overflow
```

**Status**: ✅ **FULLY VALIDATED** - Handles long-running incidents with proper timeout, compression, and state management.

---

## Lesson 7: Choose Agent Based on Need ⚠️ PARTIALLY VALIDATED

**Principle**: Don't use multi-agent when simple function call suffices - sometimes a function is better.

### How This Project Does It:

| Aspect | Implementation | Status |
|--------|---------------|--------|
| **Simple Functions** | `extract_identifiers()` - pure regex | ✅ Good |
| **Discovery Functions** | `get_services_impl()`, `discover_jaeger_services_impl()` | ✅ Good |
| **Validation Functions** | `validate_sql_query()` - pure function | ✅ Good |
| **Heavy CrewAI** | Sometimes complex when simple would do | ⚠️ Over-engineered |

```python
# GOOD: Simple function for regex extraction
def extract_identifiers(text: str, payload: Dict) -> Dict[str, str]:
    # Pure function - no need for agent
    identifiers = {}
    ucic_patterns = [r'\bUCIC[:\s]*(\d{10,12})\b', ...]
    # Regex extraction - fast and deterministic
    return identifiers

# COULD BE SIMPLER: Some tasks use full Crew when not needed
# For example: format_incidents_for_llm() could be simple string formatting
```

**Status**: ⚠️ **MIXED** - Project does use simple functions where appropriate, but could simplify some LLM calls.

---

# 📊 Summary Table

| Lesson | Status | Notes |
|--------|--------|-------|
| **Lesson 1: Max Leverage** | ✅ FULLY VALIDATED | Complex multi-agent reasoning |
| **Lesson 2: Craftsmanship** | ✅ FULLY VALIDATED | Detailed prompts, validation |
| **Lesson 3: No Bad Shipping** | ⚠️ PARTIALLY | Safety exists, need more testing |
| **Lesson 4: Source + Autoloops** | ✅ FULLY VALIDATED | Real data sources, 5 autoloops |
| **Lesson 5: (N/A)** | - | Not applicable |
| **Lesson 6: Long-Running** | ✅ FULLY VALIDATED | 4hr timeout, compression |
| **Lesson 7: Choose Wisely** | ⚠️ PARTIALLY | Good mix, some over-engineering |

---

# 🎯 Areas for Improvement

### 1. Lesson 3: Better Testing Needed
```
current:  Some test files exist
needed:  - Prompt A/B testing
         - Integration tests
         - Human review gates
```

### 2. Lesson 7: Simplify Some Agents
```
current:  Some simple tasks use full CrewAI
could be: - format_incidents_for_llm → simple function
         - Some summary tasks → direct LLM call
```

---

This project is a **well-architected AI system** that validates against most lessons, with strong autoloops (Lesson 4) and long-running support (Lesson 6). The main gaps are in testing (Lesson 3) and occasionally over-using agents where simple functions would suffice (Lesson 7).
