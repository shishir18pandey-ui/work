import os

from pydantic import BaseModel, Field
from typing import Any, Type
from crewai.tools import BaseTool
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from elasticsearch import AsyncElasticsearch
from datetime import datetime, timedelta
import logging

CA_CERT_FILE = os.getenv("CA_CERT_FILE", "./IDFCBANKCA.pem")

logger = logging.getLogger(__name__)

class ElasticsearchSearchToolInput(BaseModel):
    search_terms: str = Field(description="Technical keywords, error codes, log levels or exceptions.")
    search_tags: dict[str, str] = Field(..., description="A dictionary of tags to search.")

class ElasticsearchSearchTool(BaseTool):
    name: str = "Elasticsearch query tool"
    description: str = "Tool to search a given elasticsearch endpoint for get documents pertaining to given inputs"
    args_schema: Type[BaseModel] = ElasticsearchSearchToolInput

    es_endpoints: list[str] = Field(..., description="Elasticsearch cluster URL")
    es_auth_header: str = Field(..., description="Base64 encoded credentials for Bearer auth")
    app_name: str = Field(..., description="The codified app name to search get indexes")

    def _run(self, search_text: str, search_tags: dict[str, str]) -> str:

        def run_in_loop():
            return asyncio.run(self._async_search(search_text, search_tags))

        try:
            asyncio.get_running_loop()

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_in_loop)
                result = future.result()
        except RuntimeError:
            result = run_in_loop()

        return json.dumps(result, indent=2)

    async def _arun(self, search_text: str, search_tags: dict[str, str]) -> Any:
        result = await self._async_search(search_text, search_tags)
        return json.dumps(result, indent=2)

    async def _async_search(self, text: str, tags: dict[str, str]):
        clients: list[dict[str, Any]] = []
        for url in self.es_endpoints:
            es = AsyncElasticsearch(
                url,
                headers={"Authorization": f"Bearer {self.es_auth_header}"},
                ca_certs=CA_CERT_FILE,
            )
            clients.append({"url": url, "client": es})

        try:
            cluster_indices = []

            async def get_indices_for_client(cluster_info):
                client: AsyncElasticsearch = cluster_info["client"]
                try:
                    res = await client.cat.indices(format="json")
                    valid_indices = [idx["index"] for idx in res if idx["index"].find(self.app_name) >= 0 and not idx["index"].startswith(".")] # type: ignore
                    if valid_indices:
                        logger.info(f"Found Valid Indices: { json.dumps(valid_indices, indent=2) }")
                        return (client, cluster_info["url"], valid_indices)
                    else:
                        logger.info(f"Unable to find valid indices - Using STANDARD filter")
                        valid_indices = ["elk-*-prod*"]
                        return (client, cluster_info["url"], valid_indices)
                except Exception:
                    return None

            indexes_task = [asyncio.create_task(get_indices_for_client(c)) for c in clients]
            results = await asyncio.gather(*indexes_task)

            for res in results:
                if res:
                    cluster_indices.append(res)

            if not cluster_indices:
                return { "error": "no valid indicies found across any of the provided Elasticsearch clusters"}

            must_clauses = [{"match": {k: v for k, v in tags.items()}}]
            base_query = {
                "query": {
                    "bool": {
                        "must": must_clauses,
                        "should": [{"multi_match": { "query": text, "fields": ["message", "log.level", "error.message", "stack_trace", "*"]}}],
                        "minimum_should_match": 1,
                        "filter": []
                    }
                }
            }

            time_buckets = [(0, 24), (24, 48), (48, 96), (96, 196)]

            for start_hrs, end_hrs in time_buckets:
                hits = await self._search_buckets_across_clusters(base_query, cluster_indices, start_hrs, end_hrs)
                if hits:
                    return {
                        "status": "success",
                        "time_bucket": f"{start_hrs} - {end_hrs} hours ago",
                        "hits": hits
                    }

            return { "status": "no_data", "message": "No logs found across all clusters and time buckets."}

        finally:
            for c in clients:
                await c["client"].close()

    async def _search_buckets_across_clusters(self, base_query: dict, valid_indices: list[tuple[AsyncElasticsearch,str,list[str]]], start_hrs: int, end_hrs: int):
        now = datetime.now()
        start_time = (now - timedelta(hours=end_hrs)).isoformat()
        end_time = (now - timedelta(hours=start_hrs)).isoformat()

        query = base_query.copy()
        query["query"]["bool"]["filter"] = [
            { "range": { "@timestamp": { "gte": start_time, "lte": end_time }}}
        ]

        async def query_index(client: AsyncElasticsearch, url: str, index: str):
            res = await client.search(index=index, body=query, size=25)
            hits = res.get("hits", {}).get("hits", [])

            for hit in hits:
                hit["_source_cluster"] = url
                logger.info(f"Hits for {index}: {json.dumps(hit)}")
            return hits

        tasks = []
        for client, url, indices in valid_indices:
            for idx in indices:
                tasks.append(asyncio.create_task(query_index(client, url, idx)))

        if not tasks:
            return []

        for task in asyncio.as_completed(tasks):
            try:
                index_hits = await task
                if index_hits:
                    for t in tasks:
                        if not t.done():
                            t.cancel()
                return index_hits
            except Exception:
                pass

        return []



