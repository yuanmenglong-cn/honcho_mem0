"""mem0 search tool for honcho agent tool loop."""

import logging
from typing import Any

from src.config import settings
from src.mem0_integration import get_mem0_client
from src.utils.agent_tools import ToolContext

logger = logging.getLogger(__name__)

SEARCH_MEM0_TOOL: dict[str, Any] = {
    "name": "search_mem0",
    "description": "Search the external mem0 memory store for relevant memories about the peer. Use this to find cross-session, long-term facts that may not be in the current workspace.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to find relevant memories",
            },
            "target_peer": {
                "type": "string",
                "description": "Optional: specific peer to filter memories for (defaults to the observed peer)",
            },
        },
        "required": ["query"],
    },
}


async def _handle_search_mem0(
    ctx: ToolContext, tool_input: dict[str, Any]
) -> str:
    """Handle search_mem0 tool call from agent tool loop."""
    if not settings.MEM0.ENABLED or not settings.MEM0.ADD_TOOL:
        return "mem0 is not enabled"

    query = tool_input.get("query", "")
    if not query:
        return "No query provided for mem0 search."

    # Determine observer and observed
    # observer = who is searching (the entity building/retrieving memories)
    observer = ctx.observer
    # observed = who we're searching memories about (defaults to ctx.observed)
    observed = tool_input.get("target_peer") or ctx.observed

    try:
        mem0_client = get_mem0_client()
        results = await mem0_client.search(
            query=query,
            workspace_name=ctx.workspace_name,
            observer=observer,  # ← 正确的 observer（谁在看）
            observed=observed,    # ← 被看的人（作为 agent_id 过滤）
        )
    except Exception as e:
        logger.debug(f"mem0 search tool failed: {e}")
        return f"mem0 search failed: {e}"

    if not results:
        return f"No memories found in mem0 for {observed} with this query."

    logger.info(
        "mem0 tool returned %d result(s) for query=%r, workspace=%r, observer=%r, observed=%r: %s",
        len(results),
        query,
        ctx.workspace_name,
        observer,
        observed,
        "; ".join(r.get("memory", "") for r in results[:5]),  # 只记录前5条
    )

    lines = [f"Memories about {observed} from mem0:"]
    for r in results:
        memory = r.get("memory", "")
        score = r.get("score", 0.0)
        source_role = (r.get("metadata") or {}).get("source_role", "")
        created_at = r.get("created_at", "")

        parts = [f"[{score:.2f}]"]
        if source_role == "user":
            parts.append(f"[{observed}]")
        elif source_role == "assistant":
            parts.append("[agent]")

        if created_at:
            ts = created_at.split(".")[0] if "." in created_at else created_at.split("+")[0]
            parts.append(f"[{ts.replace('T', ' ')}]")

        tag = " ".join(parts)
        lines.append(f"- {tag} {memory}")

    return "\n".join(lines)
