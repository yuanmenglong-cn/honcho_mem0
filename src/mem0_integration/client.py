"""Mem0 client wrapper with error handling and ID mapping.

Supports both mem0 cloud (via mem0ai SDK) and self-hosted mem0
(via direct HTTP calls to the mem0 server API).
"""

import logging
from functools import cache
from typing import Any

import httpx

from src.config import settings

logger = logging.getLogger(__name__)


def build_mem0_user_id(
    workspace_name: str,
    observer: str,
    prefix: str | None = None,
) -> str:
    """Build mem0 user_id from honcho workspace/observer."""
    p = prefix or settings.MEM0.USER_ID_PREFIX
    return f"{p}:{workspace_name}:{observer}"


class _SelfHostedMem0Client:
    """Direct HTTP client for self-hosted mem0 instances."""

    def __init__(self, api_url: str, api_key: str | None, timeout: float) -> None:
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    async def add(
        self,
        messages: list[dict[str, str]],
        user_id: str,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        """Add memories via POST /memories."""
        # mem0 add API requires user_id at top level, not in filters
        payload: dict[str, Any] = {
            "messages": messages,
            "user_id": user_id,
        }
        if agent_id:
            payload["agent_id"] = agent_id

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._api_url}/memories",
                json=payload,
                headers=self._headers(),
            )
            data = resp.json()
            # mem0 may return Neo4j errors as 200 OK with error in JSON body
            if "detail" in data and data.get("results", []) == []:
                raise httpx.HTTPStatusError(
                    f"mem0 add returned error: {data['detail']}",
                    request=resp.request,
                    response=resp,
                )
            return data

    async def search(
        self,
        query: str,
        user_id: str,
        agent_id: str | None = None,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Search memories via POST /search."""
        # New mem0 API: user_id and agent_id must be in filters, not top-level
        payload: dict[str, Any] = {"query": query}

        # Build filters dict with user_id and agent_id
        filters_dict: dict[str, Any] = filters or {}
        filters_dict["user_id"] = user_id
        if agent_id:
            filters_dict["agent_id"] = agent_id
        payload["filters"] = filters_dict

        if top_k is not None:
            payload["top_k"] = top_k
        if threshold is not None:
            payload["threshold"] = threshold

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._api_url}/search",
                json=payload,
                headers=self._headers(),
            )
            resp.raise_for_status()
            return resp.json()

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self._api_key:
            h["X-API-Key"] = self._api_key
        return h


class Mem0Client:
    """Wrapper around mem0 with support for cloud and self-hosted modes."""

    def __init__(
        self,
        api_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._api_url: str = api_url or settings.MEM0.API_URL
        self._api_key: str | None = api_key or settings.MEM0.API_KEY
        self._timeout: float = settings.MEM0.TIMEOUT_SECONDS
        self._enabled: bool = settings.MEM0.ENABLED
        self._sdk_client: Any | None = None
        self._self_hosted: _SelfHostedMem0Client | None = None
        self._mode: str = "none"  # "sdk", "self_hosted", or "none"

    def _ensure_client(self) -> str:
        """Lazily initialize either SDK or self-hosted client.

        Returns the active mode: "sdk", "self_hosted", or "none".
        """
        if not self._enabled:
            self._mode = "none"
            return "none"
        if self._mode != "none":
            return self._mode

        # Try self-hosted first (direct HTTP)
        try:
            self._self_hosted = _SelfHostedMem0Client(
                api_url=self._api_url,
                api_key=self._api_key,
                timeout=self._timeout,
            )
            self._mode = "self_hosted"
            return "self_hosted"
        except Exception as e:
            logger.debug(f"Failed to initialize self-hosted mem0: {e}")

        # Fall back to SDK (cloud mode)
        try:
            from mem0 import AsyncMemoryClient  # pyright: ignore[reportMissingImports]

            self._sdk_client = AsyncMemoryClient(
                api_key=self._api_key or "dummy",
                host=self._api_url,
            )
            inner = getattr(self._sdk_client, "async_client", None)
            if inner is not None and hasattr(inner, "timeout"):
                inner.timeout = httpx.Timeout(self._timeout)
            self._mode = "sdk"
            return "sdk"
        except Exception as e:
            logger.debug(f"Failed to initialize mem0 SDK client: {e}")
            self._mode = "none"
            return "none"

    async def search(
        self,
        query: str,
        workspace_name: str,
        observer: str,
        observed: str | None = None,
        agent_id: str | None = None,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Search mem0 for relevant memories. Returns empty list on error.

        Handles both cloud SDK results (``results`` key) and self-hosted
        graph-based mem0 results (``relations`` key).

        Args:
            query: Search query string
            workspace_name: Honcho workspace name
            observer: Observer peer name
            observed: Observed peer name
            agent_id: Optional mem0 agent_id to filter by
            filters: Optional mem0 filters dict (e.g. {"source_role": "user"})
            top_k: Maximum number of results to return
            threshold: Minimum similarity score for results
        """
        if not self._enabled:
            return []

        mode = self._ensure_client()
        if mode == "none":
            return []

        try:
            user_id = build_mem0_user_id(workspace_name, observer)
            if mode == "self_hosted" and self._self_hosted:
                result = await self._self_hosted.search(
                    query=query,
                    user_id=user_id,
                    agent_id=agent_id,
                    filters=filters,
                    top_k=top_k,
                    threshold=threshold,
                )
                if result is None:
                    return []
                memories = result.get("results") or []
                relations = result.get("relations") or []
                if memories:
                    logger.info(
                        "mem0 search returned %d result(s) for query=%r, user_id=%r, agent_id=%r: %s",
                        len(memories),
                        query,
                        user_id,
                        agent_id or "",
                        "; ".join(
                            f"{r.get('memory', '')} (score={r.get('score', 0):.2f}, "
                            f"role={(r.get('metadata') or {}).get('source_role', '')}, "
                            f"at={r.get('created_at', '')})"
                            for r in memories
                        ),
                    )
                    return memories
                if relations:
                    fallback = [
                        {"memory": f"{r.get('source', '')} {r.get('relationship', '')} {r.get('target', '')}",
                         "metadata": {},
                         "score": 0.0,
                         "created_at": ""}
                        for r in relations
                    ]
                    logger.info(
                        "mem0 search returned %d relation(s) for user_id=%r, agent_id=%r: %s",
                        len(relations),
                        user_id,
                        agent_id or "",
                        "; ".join(
                            f"{r.get('source', '')} {r.get('relationship', '')} {r.get('target', '')}"
                            for r in relations
                        ),
                    )
                    return fallback
                return []

            # SDK mode
            if self._sdk_client:
                kwargs: dict[str, Any] = {"user_id": user_id}
                if agent_id:
                    kwargs["agent_id"] = agent_id
                if filters:
                    kwargs["filters"] = filters
                response = await self._sdk_client.search(query, **kwargs)
                if response is None:
                    return []
                sdk_results = response.get("results") or []
                logger.info(
                    "mem0 search returned %d result(s) for query=%r, user_id=%r, agent_id=%r: %s",
                    len(sdk_results),
                    query,
                    user_id,
                    agent_id or "",
                    "; ".join(
                        f"{r.get('memory', '')} (score={r.get('score', 0):.2f}, "
                        f"role={(r.get('metadata') or {}).get('source_role', '')})"
                        for r in sdk_results
                    ),
                )
                return sdk_results
        except httpx.ConnectError as e:
            logger.warning(
                "mem0 search failed: connection error to %s: %s. "
                "Is the mem0 server running at %s?",
                self._api_url, e, self._api_url
            )
        except httpx.TimeoutException as e:
            logger.warning(
                "mem0 search failed: timeout after %.1fs to %s: %s",
                self._timeout, self._api_url, e
            )
        except httpx.HTTPStatusError as e:
            resp = e.response
            body = getattr(resp, "text", "")[:200]
            logger.warning(
                "mem0 search failed: status=%s, url=%s, error=%s, body=%s",
                resp.status_code, e.request.url, e, body[:100],
            )
        except httpx.HTTPError as e:
            resp = getattr(e, "response", None)
            status = "N/A"
            if resp:
                try:
                    status = resp.status_code
                except Exception:
                    status = "N/A"
            url = "N/A"
            try:
                req = getattr(e, "request", None)
                if req:
                    url = str(req.url)
            except Exception:
                pass
            body = ""
            if resp:
                try:
                    body = getattr(resp, "text", "")[:200]
                except Exception:
                    body = ""
            logger.warning(
                "mem0 search failed: status=%s, url=%s, error=%s, body=%s",
                status, url, e, body[:100],
            )
        except Exception as e:
            logger.warning(f"mem0 search unexpected error: {e}")
        return []

    async def add(
        self,
        messages: list[dict[str, str]],
        workspace_name: str,
        observer: str,
        observed: str | None = None,
        agent_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Add memories to mem0. Returns empty list on error."""
        if not self._enabled:
            return []

        mode = self._ensure_client()
        if mode == "none":
            return []

        try:
            user_id = build_mem0_user_id(workspace_name, observer)
            target_agent_id = agent_id or observed

            if mode == "self_hosted" and self._self_hosted:
                result = await self._self_hosted.add(
                    messages=messages,
                    user_id=user_id,
                    agent_id=target_agent_id,
                )
                results = result.get("results", [])
                # Log detailed results including what was stored
                for i, r in enumerate(results[:3]):  # Log first 3 results
                    logger.info(
                        "mem0 stored memory [%d/%d]: memory=%r, event=%r, id=%r",
                        i + 1,
                        len(results),
                        r.get("memory", "")[:100],
                        r.get("event", "unknown"),
                        r.get("id", "N/A"),
                    )
                logger.info(
                    "mem0 add returned %d result(s) for user_id=%r, agent_id=%r",
                    len(results),
                    user_id,
                    target_agent_id or "",
                )
                return results

            # SDK mode
            if self._sdk_client:
                kwargs: dict[str, Any] = {"user_id": user_id}
                if target_agent_id:
                    kwargs["agent_id"] = target_agent_id
                response = await self._sdk_client.add(messages, **kwargs)
                results = response.get("results", [])
                # Log detailed results including what was stored
                for i, r in enumerate(results[:3]):  # Log first 3 results
                    logger.info(
                        "mem0 stored memory [%d/%d]: memory=%r, event=%r, id=%r",
                        i + 1,
                        len(results),
                        r.get("memory", "")[:100],
                        r.get("event", "unknown"),
                        r.get("id", "N/A"),
                    )
                logger.info(
                    "mem0 add returned %d result(s) for user_id=%r, agent_id=%r",
                    len(results),
                    user_id,
                    target_agent_id or "",
                )
                return results
        except httpx.ConnectError as e:
            logger.warning(
                "mem0 add failed: connection error to %s: %s. "
                "Is the mem0 server running at %s?",
                self._api_url, e, self._api_url
            )
        except httpx.TimeoutException as e:
            # Log the messages that failed due to timeout
            msg_preview = [m.get("content", "")[:50] + "..." if len(m.get("content", "")) > 50 else m.get("content", "") for m in messages[:2]]
            logger.warning(
                "mem0 add failed: timeout after %.1fs to %s: %s. "
                "Failed to sync %d messages. Preview: %r",
                self._timeout, self._api_url, e, len(messages), msg_preview
            )
        except httpx.HTTPStatusError as e:
            resp = e.response
            body = getattr(resp, "text", "")[:200]
            logger.warning(
                "mem0 add failed: status=%s, url=%s, error=%s, body=%s",
                resp.status_code, e.request.url, e, body[:100],
            )
        except httpx.HTTPError as e:
            resp = getattr(e, "response", None)
            status = "N/A"
            if resp:
                try:
                    status = resp.status_code
                except Exception:
                    status = "N/A"
            url = "N/A"
            try:
                req = getattr(e, "request", None)
                if req:
                    url = str(req.url)
            except Exception:
                pass
            body = ""
            if resp:
                try:
                    body = getattr(resp, "text", "")[:200]
                except Exception:
                    body = ""
            logger.warning(
                "mem0 add failed: status=%s, url=%s, error=%s, body=%s",
                status, url, e, body[:100],
            )
        except Exception as e:
            logger.warning(f"mem0 add unexpected error: {e}")
        return []


@cache
def get_mem0_client() -> Mem0Client:
    """Get the mem0 client singleton (service-locator pattern)."""
    return Mem0Client()


def close_mem0_client() -> None:
    """Close the mem0 client and clear the singleton cache."""
    get_mem0_client.cache_clear()
