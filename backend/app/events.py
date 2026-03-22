"""
events.py — Structured event factory for the AgentForge pipeline.

PURPOSE:
    Every agent needs to report what it's doing. These reports serve:
    1. Terminal output (Day 1) — so you see progress while it runs
    2. WebSocket streaming (Day 2) — so the frontend can animate the graph
    3. Debugging — so you can trace exactly what happened in any run

WHY A FACTORY FUNCTION?
    Without this, each agent would build event dicts manually:
    
        # In orchestrator.py:
        event = {"type": "agent_event", "agent": "orchestrator", ...}
        
        # In researcher.py:
        event = {"type": "agent_event", "agent": "researcher", ...}
    
    Every agent would repeat the same structure, and inevitably someone
    would typo a key name ("stauts" instead of "status") causing a bug
    that only shows up in the frontend.
    
    A factory function enforces a consistent shape. If we ever need to
    add a field (like "duration_ms"), we add it in ONE place.

DESIGN PATTERN:
    This is the Factory Pattern — one of the simplest and most useful
    creational patterns. The function encapsulates object creation logic
    so callers don't need to know the internal structure.

EVENT SCHEMA:
    {
        "type": "agent_event",       # Always "agent_event" (for filtering)
        "agent": "researcher",       # Which agent emitted this
        "status": "running",         # running | completed | error
        "message": "Searching...",   # Human-readable description
        "timestamp": "2026-...",     # ISO 8601 UTC timestamp
        "data": { ... }              # Optional payload (flexible)
    }
"""

from datetime import datetime, timezone
from typing import Optional


def make_event(
    agent: str,
    status: str,
    message: str,
    data: Optional[dict] = None,
) -> dict:
    """
    Create a structured agent event.
    
    Args:
        agent:   Name of the emitting agent.
                 One of: "orchestrator", "researcher", "writer", "critic", "system"
                 
        status:  Current status of the agent.
                 One of:
                   - "running"   → agent is actively working
                   - "completed" → agent finished successfully
                   - "error"     → something went wrong
                   
        message: Human-readable description of what's happening.
                 Should be specific enough to be useful in logs.
                 Good:  "Searching for: 'quantum computing 2026'"
                 Bad:   "Working..."
                 
        data:    Optional dict with structured payload.
                 Examples:
                   - Orchestrator: {"plan": [...]}
                   - Researcher:   {"sources_count": 5, "sources": [...]}
                   - Critic:       {"review": {"score": 8, ...}}
                 Default is an empty dict (not None) to simplify
                 downstream code that iterates over data.keys().
    
    Returns:
        A dict matching the event schema described above.
    
    Example:
        >>> event = make_event(
        ...     agent="researcher",
        ...     status="running",
        ...     message="Searching for: 'quantum computing breakthroughs'",
        ...     data={"search_query": "quantum computing breakthroughs"},
        ... )
        >>> event["agent"]
        'researcher'
        >>> event["type"]
        'agent_event'
    """
    return {
        "type": "agent_event",
        
        "agent": agent,
        # WHY NOT AN ENUM?
        # We could define AgentName = Literal["orchestrator", "researcher", ...]
        # but that adds rigidity. If someone adds a new agent, they'd need
        # to update the enum AND this file. A plain string is more flexible
        # for a project this size. In a team of 10+, use an enum.
        
        "status": status,
        # Same reasoning — Literal type would be stricter but adds friction.
        
        "message": message,
        
        "timestamp": datetime.now(timezone.utc).isoformat(),
        # WHY UTC?
        # Always store/transmit timestamps in UTC. The frontend converts
        # to local time for display. This avoids timezone bugs when
        # server and client are in different timezones (e.g., your server
        # is on Railway in US-East, but you're in Jeddah UTC+3).
        #
        # WHY .isoformat()?
        # ISO 8601 ("2026-03-19T14:30:00+00:00") is the universal standard
        # for timestamp serialization. JavaScript's new Date() parses it
        # natively. Every language/tool understands it.
        
        "data": data if data is not None else {},
        # WHY DEFAULT TO {} INSTEAD OF None?
        # Downstream code can always do event["data"].get("key") without
        # first checking if data is None. Eliminates a class of KeyError bugs.
    }