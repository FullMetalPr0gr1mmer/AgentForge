"""
state.py — The shared state schema for the AgentForge pipeline.

WHY THIS IS THE MOST IMPORTANT FILE:
    In LangGraph, the state is the ONLY way agents communicate. There is
    no direct agent-to-agent messaging. Instead:
    
    1. Agent A reads from the state
    2. Agent A returns a partial update to the state
    3. LangGraph merges that update into the current state
    4. Agent B reads the updated state
    
    This file defines the SHAPE of that state — what fields exist, what
    types they hold, and how updates get merged.

HOW LANGGRAPH STATE UPDATES WORK:
    When an agent node returns {"draft": "some text", "agent_logs": [event]},
    LangGraph doesn't replace the entire state. It merges field by field:
    
    - For most fields: the new value REPLACES the old value.
      e.g., returning {"draft": "v2"} overwrites the old draft.
    
    - For fields with a REDUCER (like agent_logs): the new value is
      COMBINED with the old value using the reducer function.
      e.g., returning {"agent_logs": [new_event]} APPENDS to existing logs.
    
    This merge behavior is why we use TypedDict + Annotated, not a plain dict.

DESIGN DECISIONS:
    - TypedDict over @dataclass: LangGraph's StateGraph expects TypedDict.
      Dataclasses would need conversion and add unnecessary friction.
    
    - Separate typed dicts for SubTask, ResearchNote, ReviewResult:
      These give us type safety AND serve as documentation. When you see
      List[SubTask] you immediately know the shape of the data, unlike
      List[dict] which tells you nothing.
    
    - Optional fields: Most fields start as None because agents populate
      them sequentially. The Orchestrator doesn't know about the draft;
      the Writer doesn't produce a review.
"""

from typing import TypedDict, List, Optional
from typing_extensions import Annotated


# ─── Reducer Function ───────────────────────────────────────
# 
# A "reducer" tells LangGraph HOW to merge a field when multiple
# updates arrive. Without a reducer, new values overwrite old ones.
# With this reducer, new items get appended to the existing list.
#
# This is critical for agent_logs: every agent appends its events,
# and we never want to lose earlier events.
#
# ANALOGY: Think of it like Git. Without a reducer, it's a force-push
# (overwrite). With a reducer, it's a merge (combine both versions).

def append_list(existing: list, new: list) -> list:
    """
    Reducer that appends new items to an existing list.
    
    LangGraph calls this as: result = append_list(current_value, returned_value)
    
    Args:
        existing: The current list in the state (could be empty [])
        new: The new list returned by an agent node
    
    Returns:
        Combined list with new items appended after existing ones
    
    Example:
        State has agent_logs = [event1, event2]
        Agent returns {"agent_logs": [event3]}
        Result: agent_logs = [event1, event2, event3]
    """
    return existing + new


# ─── Data Shapes ────────────────────────────────────────────
#
# These TypedDicts define the structure of complex nested data.
# They're NOT LangGraph state — they're building blocks used
# INSIDE the state.

class SubTask(TypedDict):
    """
    A single research subtask produced by the Orchestrator.
    
    Example:
        {
            "id": 1,
            "description": "Find recent advances in quantum error correction",
            "search_query": "quantum error correction breakthroughs 2025 2026"
        }
    """
    id: int                 # Sequential ID (1, 2, 3, ...)
    description: str        # What this subtask should investigate
    search_query: str       # Optimized query for the search API


class ResearchNote(TypedDict):
    """
    Research findings for a single subtask, produced by the Researcher.
    
    Example:
        {
            "task_id": 1,
            "content": "Google announced a breakthrough in...",
            "sources": ["https://blog.google/...", "https://arxiv.org/..."]
        }
    """
    task_id: int            # Links back to SubTask.id
    content: str            # Summarized findings as text
    sources: List[str]      # URLs of sources used


class ReviewResult(TypedDict):
    """
    Quality evaluation produced by the Critic agent.
    
    The score field drives the conditional routing in the graph:
    - score >= QUALITY_THRESHOLD → accept draft, go to finalize
    - score < QUALITY_THRESHOLD → send back to Writer for revision
    
    Example:
        {
            "score": 6,
            "strengths": ["Well-organized", "Good use of sources"],
            "weaknesses": ["Missing recent 2026 developments"],
            "suggestions": ["Add a section on hardware advances"]
        }
    """
    score: int              # 1-10 quality rating
    strengths: List[str]    # What the draft does well
    weaknesses: List[str]   # What needs improvement
    suggestions: List[str]  # Specific actionable fixes


# ─── The Main State ─────────────────────────────────────────
#
# This is what flows through the entire LangGraph pipeline.
# Every agent node receives this as input and returns a partial
# update to it.

class AgentState(TypedDict):
    """
    The complete state of an AgentForge pipeline run.
    
    LIFECYCLE:
        1. User submits query → state initialized with query + defaults
        2. Orchestrator reads query → writes plan
        3. Researcher reads plan → writes research_notes
        4. Writer reads query + research_notes → writes draft
        5. Critic reads draft + research_notes → writes review
        6. If review.score < threshold: back to step 4 (revision)
        7. Finalize reads draft → writes final_output
    
    FIELD CATEGORIES:
        - Input:    query (set once, never changes)
        - Pipeline: plan, research_notes, draft, review (set by agents)
        - Control:  revision_count (tracks loop iterations)
        - Output:   final_output (the deliverable)
        - Logging:  agent_logs (append-only event stream)
    """
    
    # ── Input ──
    query: str
    # The user's original research question.
    # Set once at the start, never modified.
    # Every agent can read this to stay aligned with the user's intent.
    
    # ── Orchestrator Output ──
    plan: Optional[List[SubTask]]
    # The decomposed research plan.
    # None until the Orchestrator runs.
    # The Researcher iterates over this to know what to search for.
    
    # ── Researcher Output ──
    research_notes: Optional[List[ResearchNote]]
    # Collected findings from web searches.
    # None until the Researcher runs.
    # The Writer uses this as source material; the Critic uses it
    # to verify the draft's accuracy.
    
    # ── Writer Output ──
    draft: Optional[str]
    # The current article draft (markdown).
    # None until the Writer's first run.
    # Gets OVERWRITTEN (not appended) on each revision — we only
    # care about the latest version.
    
    # ── Critic Output ──
    review: Optional[ReviewResult]
    # The quality evaluation of the current draft.
    # None until the Critic's first run.
    # Drives the conditional routing: revise or accept.
    
    # ── Control Flow ──
    revision_count: int
    # How many times the Writer has revised the draft.
    # Starts at 0. Incremented by the Critic after each review.
    # When this hits MAX_REVISIONS, we accept whatever we have
    # (prevents infinite loops).
    
    # ── Final Output ──
    final_output: Optional[str]
    # The approved, final article. Set by the finalize node.
    # This is what gets returned to the user.
    
    # ── Event Log (append-only) ──
    agent_logs: Annotated[List[dict], append_list]
    # A chronological stream of events from all agents.
    #
    # WHY Annotated[..., append_list]?
    #   This tells LangGraph to use our append_list reducer when
    #   merging updates to this field. Without it, each agent's
    #   logs would OVERWRITE the previous agent's logs.
    #
    #   With the reducer:
    #     Orchestrator returns {"agent_logs": [event1, event2]}
    #     → state.agent_logs = [event1, event2]
    #     Researcher returns {"agent_logs": [event3]}
    #     → state.agent_logs = [event1, event2, event3]  (appended!)
    #
    # Each event is a dict matching the shape from events.py.
    # This log is streamed to the frontend via WebSocket (Day 2).