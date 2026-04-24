"""Dialectic enrichment: parallel mem0 search during prefetch."""

import logging
import math
from typing import Any

from src.config import settings
from src.mem0_integration import get_mem0_client

logger = logging.getLogger(__name__)

_DEDUP_CHARS = 100


async def enrich_representation_with_mem0(
    representation_markdown: str,
    query: str,
    workspace_name: str,
    observer: str,
    observed: str,
) -> str:
    """Enrich representation markdown with mem0 search results.

    Used by the representation endpoint when search_query is provided.
    If mem0 is disabled, returns empty results, or errors,
    returns the original representation unchanged.
    """
    if not settings.MEM0.ENABLED or not settings.MEM0.ENRICH_DIALECTIC:
        logger.debug("mem0 representation enrichment skipped: MEM0 not enabled or ENRICH_DIALECTIC=false")
        return representation_markdown

    try:
        mem0_client = get_mem0_client()
        mem0_results = await mem0_client.search(
            query=query,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            agent_id=observed,
            top_k=settings.MEM0.MAX_RESULTS,
            threshold=settings.MEM0.MIN_SCORE,
        )
    except Exception as e:
        logger.warning(f"mem0 representation search failed: {e}")
        return representation_markdown

    if not mem0_results:
        logger.info(
            "mem0 representation search returned no results for query=%r, workspace=%r, observer=%r, observed=%r",
            query,
            workspace_name,
            observer,
            observed,
        )
        return representation_markdown

    logger.info(
        "mem0 representation search returned %d result(s) for query=%r, workspace=%r, observer=%r, observed=%r: %s",
        len(mem0_results),
        query,
        workspace_name,
        observer,
        observed,
        "; ".join(r.get("memory", "") for r in mem0_results),
    )

    mem0_results = await _dedup_memories(mem0_results, representation_markdown)

    mem0_items = _format_mem0_items(mem0_results, observed)
    if not mem0_items:
        return representation_markdown

    mem0_section = (
        "\n\n## External Memory (mem0)\n\n"
        + "The following memories were retrieved from the external memory store:\n\n"
        + "\n".join(mem0_items)
    )

    return representation_markdown + mem0_section


async def enrich_prefetched_observations(
    original_observations: str,
    query: str,
    workspace_name: str,
    observer: str,
    observed: str,
) -> str:
    """Enrich prefetched observations with mem0 search results.

    If mem0 is disabled, returns empty results, or errors,
    returns the original observations unchanged.
    """
    if not settings.MEM0.ENABLED or not settings.MEM0.ENRICH_DIALECTIC:
        return original_observations

    try:
        mem0_client = get_mem0_client()
        mem0_results = await mem0_client.search(
            query=query,
            workspace_name=workspace_name,
            observer=observed,
            top_k=settings.MEM0.MAX_RESULTS,
            threshold=settings.MEM0.MIN_SCORE,
        )
    except Exception as e:
        logger.debug(f"mem0 enrichment search failed: {e}")
        return original_observations

    if not mem0_results:
        return original_observations

    logger.info(
        "mem0 enrichment returned %d result(s) for query=%r, workspace=%r, observed=%r: %s",
        len(mem0_results),
        query,
        workspace_name,
        observed,
        "; ".join(r.get("memory", "") for r in mem0_results),
    )

    mem0_results = await _dedup_memories(mem0_results, original_observations)

    mem0_items = _format_mem0_items(mem0_results, observed)
    if not mem0_items:
        return original_observations

    mem0_section = (
        "\n\n## External Memory (mem0)\n\n"
        + "The following memories were retrieved from the external memory store:\n\n"
        + "\n".join(mem0_items)
    )

    return original_observations + mem0_section


def _format_mem0_items(mem0_results: list[dict[str, Any]], observed: str) -> list[str]:
    """Format mem0 results as markdown list items."""
    items: list[str] = []
    for result in mem0_results:
        memory = result.get("memory", "")
        if not memory:
            continue
        score = result.get("score", 0.0)
        speaker = _format_speaker(result, observed)
        timestamp = _format_timestamp(result)
        tag = f"[mem0][{score:.2f}]"
        if speaker:
            tag += f" [{speaker}]"
        if timestamp:
            tag += f" [{timestamp}]"
        items.append(f"- {tag} {memory}")
    return items


async def _dedup_memories(
    mem0_results: list[dict[str, Any]],
    existing_markdown: str,
) -> list[dict[str, Any]]:
    """Remove mem0 results that duplicate existing observations.

    Uses a fast path with strict prefix matching, then falls back to
    embedding cosine similarity for semantic deduplication.
    """
    existing_prefixes = _extract_content_prefixes(existing_markdown)

    # Fast path: strict prefix dedup
    candidates = [r for r in mem0_results if not _is_duplicate(r.get("memory", ""), existing_prefixes)]
    if not candidates:
        return []

    # Embedding-based semantic dedup
    try:
        from src.embedding_client import embedding_client

        import re
        existing_lines = []
        for line in existing_markdown.split("\n"):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("- ") or stripped.startswith("* "):
                stripped = stripped[2:].strip()
            # Strip leading timestamp like [2026-04-13 00:19:52]
            stripped = re.sub(r"^\[\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\]\s*", "", stripped)
            if stripped:
                existing_lines.append(stripped)
        if not existing_lines:
            return candidates

        candidate_texts = [r.get("memory", "") for r in candidates]
        all_texts = existing_lines + candidate_texts
        embeddings = await embedding_client.simple_batch_embed(all_texts)

        existing_embs = embeddings[: len(existing_lines)]
        candidate_embs = embeddings[len(existing_lines) :]

        threshold = settings.MEM0.DEDUP_SIMILARITY_THRESHOLD
        kept: list[dict[str, Any]] = []
        for cand, cand_emb in zip(candidates, candidate_embs):
            is_dup = any(
                _cosine_similarity(cand_emb, exist_emb) >= threshold
                for exist_emb in existing_embs
            )
            if not is_dup:
                kept.append(cand)
        return kept
    except Exception as e:
        logger.debug(f"Embedding dedup failed, returning candidates: {e}")
        return candidates


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _format_speaker(result: dict[str, Any], observed: str) -> str:
    """Extract speaker identity from mem0 result metadata."""
    metadata: dict[str, Any] = result.get("metadata", {}) or {}
    source_role = metadata.get("source_role", "")
    if source_role == "user":
        return observed
    elif source_role == "assistant":
        return "agent"
    return ""


def _format_timestamp(result: dict[str, Any]) -> str:
    """Extract and format timestamp from mem0 result metadata."""
    created_at = result.get("created_at", "")
    if not isinstance(created_at, str) or not created_at:
        return ""
    # "2026-04-15T07:40:30.849091+00:00" → "2026-04-15 07:40:30"
    ts = created_at.split(".")[0] if "." in created_at else created_at.split("+")[0]
    return ts.replace("T", " ")


def _extract_content_prefixes(text: str) -> list[str]:
    """Extract content prefixes from observation text for deduplication."""
    prefixes: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("- ") or stripped.startswith("* "):
            content = stripped[2:].strip()
            if content:
                prefixes.append(content[:_DEDUP_CHARS])
    return prefixes


def _is_duplicate(content: str, existing_prefixes: list[str]) -> bool:
    """Check if content is a duplicate of any existing observation."""
    content_prefix = content[:_DEDUP_CHARS]
    return any(content_prefix == prefix for prefix in existing_prefixes)
