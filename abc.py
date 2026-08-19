import os
import time
import logging

os.environ["OTEL_SDK_DISABLED"] = "true"

logger = logging.getLogger(__name__)


def format_incidents_for_llm(results, max_conversation_chars=100000):
    if not results:
        return "No similar historic incidents found."

    formatted = []

    for i, inc in enumerate(results, 1):
        if 'document' in inc:
            doc = inc.get('document', {})
            incident_id = doc.get('id', 'N/A')
            content = doc.get('content', '')
            chunked_content = inc.get('chunked_content', content)

            score = inc.get('score') or inc.get('dense_score') or 0

            metadata = inc.get('metadata', {})
            assignment_group = metadata.get('assignment_group', 'N/A')

            if len(chunked_content) > max_conversation_chars:
                chunked_content = chunked_content[:max_conversation_chars] + "..."

            resolution = "No resolution notes available"
            if 'Ticket Resolution notes is:' in content:
                try:
                    resolution_start = content.find('Ticket Resolution notes is:')
                    resolution_end = content.find('\n', resolution_start + 30)
                    if resolution_end == -1:
                        resolution = content[resolution_start + 30:resolution_start + 500]
                    else:
                        resolution = content[resolution_start + 30:resolution_end].strip()
                except Exception:
                    pass

            incident_str = f"""
=== SIMILAR INCIDENT {i} (Similarity: {score:.2%}) ===
Incident ID: {incident_id}
Assignment Group: {assignment_group}

CONTENT:
{chunked_content}

RESOLUTION:
{resolution}

---
"""
        else:
            incident_str = f""" Document not available. """
        formatted.append(incident_str)

    return "\n".join(formatted)


async def run_incident_context_crew_async(incident_description: str, application: str, top_k: int = 5) -> str:
    from crewai import Agent, Task, Crew, Process, LLM
    from typing import List, Dict
    from new_flow.utils.http_calls import http_client_post_async
    from new_flow.utils.llm import OPENAI_MODEL_NAME, llm_config
    from utils.observability import get_tracer

    logger = logging.getLogger(__name__)

    semantic_search_endpoint = os.getenv("SEMANTIC_SEARCH_ENDPOINT")

    async def similarity_search_api_async(index: str, query_string: str, application: str) -> List[Dict]:
        headers = {
            "Content-Type": "application/json",
        }
        body = {
            "metadata": {"application": application},
            "query": query_string,
            "index_source": index
        }

        logger.info(
            f"[SemanticSearch] REQUEST START | endpoint={semantic_search_endpoint} "
            f"application={application} index={index} query_len={len(query_string)}"
        )
        start_t = time.monotonic()

        try:
            response = await http_client_post_async(
                semantic_search_endpoint,
                headers=headers,
                json=body
            )
        except Exception as e:
            elapsed = time.monotonic() - start_t
            logger.error(
                f"[SemanticSearch] REQUEST FAILED | endpoint={semantic_search_endpoint} "
                f"elapsed={elapsed:.2f}s exception_type={type(e).__name__} "
                f"message='{str(e)}'",
                exc_info=True
            )
            raise

        elapsed = time.monotonic() - start_t
        logger.info(
            f"[SemanticSearch] REQUEST DONE | endpoint={semantic_search_endpoint} "
            f"elapsed={elapsed:.2f}s status_code={response.status_code}"
        )

        json_resp: Dict = response.json
        results: List[Dict] = []
        if json_resp:
            for obj in json_resp:
                results.append(obj)

        logger.info(
            f"[SemanticSearch] PARSED | elapsed={elapsed:.2f}s "
            f"result_count={len(results)} raw_type={type(json_resp).__name__}"
        )

        return results

    tracer = get_tracer(__name__)

    with tracer.start_as_current_span("_run_incident_context_crew_async") as span:
        span.set_attribute("incident_description", incident_description[:100] + "..." if len(incident_description) > 100 else incident_description)
        span.set_attribute("application", application)

        # ── FIX: call the search API directly, deterministically ──
        # Previously an LLM agent "decided" whether to call this via tool-calling,
        # which failed silently when MiniMax leaked a malformed tool-call and the
        # recovery patch couldn't re-invoke it — producing a fabricated summary
        # instead of real search results. Removing that dependency entirely.
        search_start = time.monotonic()
        try:
            results = await similarity_search_api_async(
                index='incidents',
                query_string=incident_description,
                application=application
            )
        except Exception as e:
            logger.error(
                f"[ContextBuilder] Search FAILED | application={application} "
                f"elapsed={time.monotonic()-search_start:.2f}s "
                f"exception_type={type(e).__name__} message='{str(e)}'",
                exc_info=True
            )
            return "No similar historic incidents found (search temporarily unavailable)."

        logger.info(
            f"[ContextBuilder] Search DONE | application={application} "
            f"elapsed={time.monotonic()-search_start:.2f}s result_count={len(results)}"
        )

        formatted_results = format_incidents_for_llm(results)

        if not results:
            return formatted_results  # "No similar historic incidents found."

        # Now just ask the LLM to summarize real, already-fetched results.
        # No tools attached — nothing left for it to "call" or leak.
        llm = LLM(
            model="openai/" + OPENAI_MODEL_NAME,
            temperature=0.0,
            base_url=llm_config.url,
            api_key=llm_config.token
        )

        agent = Agent(
            role="Historic Incident Analyst",
            goal="Summarize similar past incidents and their resolutions",
            backstory=(
                "You are a senior incident analyst. You have already been given a set of "
                "similar historic incidents retrieved from the database. Your job is only "
                "to analyze and summarize them — you do not need to search for anything yourself."
            ),
            tools=[],
            verbose=False,
            allow_delegation=False,
            llm=llm,
            temperature=0,
            max_iter=2,
            reasoning=False,
            max_retry_limit=1
        )

        task = Task(
            name="Summarize similar incidents",
            description=(
                f"Current incident:\n\n{incident_description}\n\n"
                f"=== HISTORIC SIMILAR INCIDENTS (already retrieved) ===\n{formatted_results}\n\n"
                "Analyze the results above and provide a summary of:\n"
                "1. The most relevant similar incidents found\n"
                "2. How those incidents were resolved\n"
                "3. Key troubleshooting steps from the conversation context\n"
                "4. Any patterns or common solutions that could apply to the current incident\n\n"
                "Only use the incidents provided above. Do not invent or assume incidents not listed."
            ),
            agent=agent,
            expected_output=(
                "A structured summary of similar historic incidents with their resolutions, "
                "ready to be used as context for resolving the current incident."
            ),
        )

        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True
        )

        crew_start = time.monotonic()
        logger.info(f"[ContextBuilder] CREW KICKOFF START | application={application}")

        with tracer.start_as_current_span("crew_kickoff_async") as crew_span:
            crew_span.set_attribute("incident_description", incident_description + "..." if len(incident_description) > 100 else incident_description)
            output = str(await crew.akickoff())

        crew_elapsed = time.monotonic() - crew_start
        logger.info(
            f"[ContextBuilder] CREW KICKOFF DONE | application={application} "
            f"elapsed={crew_elapsed:.2f}s output_len={len(output)}"
        )

        return output
