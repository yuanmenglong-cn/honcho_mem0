"""Deriver sync: sync raw messages to mem0 for independent processing.

Calls mem0 add once per observer, with all messages stored under
agent_id=observed. Peer classification is based on metadata (extensible):
  - type="human" → user in mem0 (role: user)
  - type="agent" → agent in mem0 (role: assistant)
Default fallback: peer_name == "owner" → human, others → agent.
"""

import asyncio
import logging
from typing import Any, Literal

from src.config import settings
from src.mem0_integration import get_mem0_client

logger = logging.getLogger(__name__)

PeerType = Literal["human", "agent"]


def _classify_peer(
    peer_name: str,
    peer_metadata: dict[str, Any] | None = None,
) -> PeerType:
    """Classify a peer as human or agent.

    Priority:
    1. metadata.type if present
    2. Fallback: "owner" → human, others → agent
    """
    if peer_metadata and "type" in peer_metadata:
        return "human" if peer_metadata["type"] == "human" else "agent"
    return "human" if peer_name == "owner" else "agent"


def _extract_timestamp(msg: Any) -> str | None:
    """Extract timestamp from a message object."""
    if getattr(msg, "created_at", None):
        ts = msg.created_at.isoformat().replace("T", " ")
        if "." in ts:
            ts = ts.split(".")[0]
        return ts
    return None


def _format_message_content(msg: Any) -> str:
    """Format message content with peer name prefix."""
    timestamp = _extract_timestamp(msg)
    if timestamp:
        return f"{msg.peer_name} (at {timestamp}): {msg.content}"
    return f"{msg.peer_name}: {msg.content}"


def _build_mem0_messages(  # pyright: ignore[reportUnusedFunction]
    messages: list[Any],
    observed: str,
    agent_name: str,
    session_name: str,
) -> list[dict[str, Any]]:
    """Build a chronological message list for a specific user-agent pair.

    Each message's content is prefixed with speaker identity and timestamp,
    so mem0's LLM infer extracts facts with speaker and temporal context.
    """
    result: list[dict[str, Any]] = []
    for msg in messages:
        if msg.peer_name == observed:
            result.append({
                "role": "user",
                "content": _format_message_content(msg),
                "metadata": {
                    "created_at": _extract_timestamp(msg),
                    "session_name": session_name,
                },
            })
        elif msg.peer_name == agent_name:
            result.append({
                "role": "assistant",
                "content": _format_message_content(msg),
                "metadata": {
                    "created_at": _extract_timestamp(msg),
                    "session_name": session_name,
                },
            })
    return result


def _split_messages_by_classification(
    messages: list[Any],
    peer_type_map: dict[str, PeerType] | None = None,
) -> tuple[list[Any], dict[str, list[Any]]]:
    """Split messages into user messages and agent message groups.

    Args:
        messages: honcho Message objects
        peer_type_map: Optional {peer_name: PeerType} map from caller.
            If None, uses default classification (owner=human, others=agent).

    Returns:
        (user_messages, {agent_name: [agent_messages]})
    """
    user_messages: list[Any] = []
    agent_messages: dict[str, list[Any]] = {}

    for msg in messages:
        msg_meta = getattr(msg, "metadata", None) or None
        peer_type = (peer_type_map or {}).get(
            msg.peer_name, _classify_peer(msg.peer_name, msg_meta)
        )
        if peer_type == "human":
            user_messages.append(msg)
        else:
            agent_messages.setdefault(msg.peer_name, []).append(msg)

    return user_messages, agent_messages


async def sync_to_mem0(
    messages: list[Any],
    workspace_name: str,
    observer: str,
    observed: str,
    session_name: str,
    peer_type_map: dict[str, PeerType] | None = None,
) -> None:
    """Sync raw messages to mem0 for a single observer.

    All messages are sent in one call with agent_id=observed,
    so the observer's mem0 namespace contains the complete
    conversation about the observed peer.

    Fire-and-forget: errors are caught and logged at DEBUG level.
    """
    logger.info(
        "mem0 sync scheduled: workspace=%r, observer=%r, observed=%r, session=%r, messages=%d",
        workspace_name,
        observer,
        observed,
        session_name,
        len(messages),
    )

    # Log self-observation case
    if observer == observed:
        logger.info(
            "mem0 sync: self-observation mode (observer=observed=%r), "
            "storing messages in own namespace",
            observer,
        )

    if not settings.MEM0.ENABLED or not settings.MEM0.SYNC_ON_DERIVE:
        logger.debug("mem0 sync skipped: MEM0 not enabled or SYNC_ON_DERIVE=false")
        return

    if not messages:
        return

    # Build all messages in chronological order (preserving original order)
    mem0_messages: list[dict[str, Any]] = []
    for msg in messages:
        msg_meta = getattr(msg, "metadata", None) or None
        peer_type = (peer_type_map or {}).get(
            msg.peer_name, _classify_peer(msg.peer_name, msg_meta)
        )
        timestamp = _extract_timestamp(msg)
        if peer_type == "human":
            mem0_messages.append({
                "role": "user",
                "content": _format_message_content(msg),
                "metadata": {
                    "created_at": timestamp,
                    "session_name": session_name,
                },
            })
        else:
            mem0_messages.append({
                "role": "assistant",
                "content": _format_message_content(msg),
                "metadata": {
                    "created_at": timestamp,
                    "session_name": session_name,
                },
            })

    # Log what we're sending to mem0
    total_chars = sum(len(m["content"]) for m in mem0_messages)
    logger.info(
        "mem0 sync payload: observer=%r, observed=%r, messages_count=%d, total_chars=%d",
        observer,
        observed,
        len(mem0_messages),
        total_chars,
    )

    try:
        mem0_client = get_mem0_client()

        result = await mem0_client.add(
            messages=mem0_messages,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            agent_id=observed,
        )
        if result:
            logger.info(
                "mem0 sync succeeded: observer=%r, observed=%r, stored_memories=%d, "
                "first_memory=%r",
                observer,
                observed,
                len(result),
                result[0].get("memory", "N/A")[:100] if result else "N/A",
            )
        else:
            logger.info(
                "mem0 sync succeeded but no memories returned: observer=%r, observed=%r",
                observer,
                observed,
            )
    except Exception as e:
        logger.warning(f"mem0 sync failed: {e}")


def schedule_sync_to_mem0(
    messages: list[Any],
    workspace_name: str,
    observer: str,
    observed: str,
    session_name: str,
    peer_type_map: dict[str, PeerType] | None = None,
) -> None:
    """Schedule mem0 sync as a fire-and-forget background task."""
    asyncio.create_task(
        sync_to_mem0(
            messages=messages,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            session_name=session_name,
            peer_type_map=peer_type_map,
        )
    )
