"""
agents/orchestrator.py — The Planner Agent.

ROLE IN THE PIPELINE:
    The Orchestrator is the FIRST agent to run. It receives the raw user
    query and produces a structured research plan — a list of 2-4 focused
    subtasks that the Researcher agent will execute.

WHY DO WE NEED THIS?
    Consider the query: "What's the state of quantum computing in 2026?"
    
    This is too broad for a single web search. You'd get surface-level
    results. But if we break it into:
      1. "Recent quantum computing hardware breakthroughs 2026"
      2. "Quantum error correction advances 2025 2026"
      3. "Major quantum computing companies funding 2026"
      4. "Quantum computing practical applications timeline"
    
    Now each search is FOCUSED and returns much better results. The
    Orchestrator is what makes this decomposition happen.

LANGGRAPH NODE CONTRACT:
    Every agent in LangGraph follows the same pattern:
    
        def node_function(state: AgentState) -> dict:
            # 1. Read what you need from state
            # 2. Do your work
            # 3. Return a PARTIAL state update (only the fields you changed)
    
    The return dict does NOT need every field from AgentState — just the
    ones this node modifies. LangGraph handles the merge.
    
    For the Orchestrator:
        Reads:   state["query"]
        Returns: {"plan": [...], "agent_logs": [...]}

PROMPT ENGINEERING NOTES:
    Getting an LLM to reliably return structured JSON is tricky. Our
    prompt uses several techniques:
    
    1. EXPLICIT FORMAT INSTRUCTION: "Respond ONLY with a valid JSON array"
       LLMs tend to add preamble ("Sure! Here's the plan:") unless you
       explicitly tell them not to.
    
    2. CONCRETE EXAMPLE: Showing an example of the exact JSON shape
       reduces format errors dramatically compared to just describing it.
    
    3. CONSTRAINTS: "2-4 subtasks" gives the LLM a range. Without this,
       you'll sometimes get 10+ subtasks (wasting API calls) or just 1
       (defeating the purpose).
    
    4. OPTIMIZED SEARCH QUERIES: We explicitly ask for "search_query"
       separate from "description" because what a human reads and what
       works well in a search engine are different things. "Find the
       latest advances in..." is a bad search query; "quantum computing
       advances 2026" is a good one.
"""

import json
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings
from app.state import AgentState
from app.events import make_event


# ─── Prompt Template ────────────────────────────────────────
#
# WHY A MODULE-LEVEL CONSTANT?
#   Prompts are configuration, not logic. Keeping them at the top of
#   the file (or in a separate prompts/ directory for larger projects)
#   makes them easy to find, read, and tweak without digging through
#   function bodies.
#
# WHY NOT LANGCHAIN'S PromptTemplate?
#   PromptTemplate adds value when you have complex variable injection,
#   partial formatting, or prompt versioning. For a single {query}
#   substitution, a plain f-string is simpler and more readable.
#   Don't add abstraction until the complexity demands it.

ORCHESTRATOR_PROMPT = """You are the Orchestrator agent in a multi-agent research system.

Your job: Take the user's query and decompose it into 2-4 focused research subtasks.

RULES:
- Each subtask should investigate a DIFFERENT aspect of the query
- Search queries should be optimized for web search (short, keyword-rich, include year if relevant)
- Return 2 subtasks for simple queries, up to 4 for complex multi-faceted ones
- Do NOT include any explanation, markdown, or preamble

For each subtask provide:
- id: sequential integer starting at 1
- description: what this subtask should investigate (1-2 sentences)
- search_query: an optimized web search query string for this subtask

Respond ONLY with a valid JSON array. Example:
[
  {{"id": 1, "description": "Find the latest breakthroughs in quantum error correction", "search_query": "quantum error correction breakthroughs 2025 2026"}},
  {{"id": 2, "description": "Identify key companies and labs leading quantum research", "search_query": "top quantum computing companies labs 2026"}}
]

User query: {query}"""

# NOTE ON DOUBLE BRACES {{ }}:
#   In Python f-strings and .format(), curly braces have special meaning.
#   To include a LITERAL { in the output, you write {{.
#   So {{"id": 1}} in the template becomes {"id": 1} in the actual prompt.
#   This is a common gotcha when writing JSON examples inside Python strings.


def _parse_llm_json(raw_text: str) -> list:
    """
    Parse JSON from LLM output, handling common quirks.
    
    LLMs often wrap JSON in markdown code fences even when you tell them
    not to. This function strips those fences before parsing.
    
    Args:
        raw_text: Raw string from the LLM response
    
    Returns:
        Parsed Python list (of subtask dicts)
    
    Raises:
        json.JSONDecodeError: If the text isn't valid JSON even after cleanup
    
    WHY A SEPARATE FUNCTION?
        1. The Critic agent also needs to parse JSON from LLMs, so we'll
           reuse this logic (DRY — Don't Repeat Yourself).
        2. It's independently testable — we can unit test JSON parsing
           without needing an LLM.
        3. It keeps the main node function focused on its core logic.
    """
    text = raw_text.strip()
    
    # Strip markdown code fences: ```json\n...\n```
    # Some models (especially Gemini) do this even when asked not to.
    if text.startswith("```"):
        # Split on first newline to remove "```json" header
        # Then remove trailing "```"
        text = text.split("\n", 1)[1]   # Everything after first line
        text = text.rsplit("```", 1)[0]  # Everything before last ```
        text = text.strip()
    
    return json.loads(text)


def orchestrator_node(state: AgentState) -> dict:
    """
    LangGraph node: Decompose the user's query into research subtasks.
    
    This function follows the LangGraph node contract:
        Input:  Full AgentState (we only read 'query')
        Output: Partial state update with 'plan' and 'agent_logs'
    
    EXECUTION FLOW:
        1. Read the query from state
        2. Send it to the LLM with our decomposition prompt
        3. Parse the JSON response into a list of SubTask dicts
        4. Handle errors gracefully (fallback to single-task plan)
        5. Return the plan + log events
    
    ERROR HANDLING PHILOSOPHY:
        We NEVER let the pipeline crash on a parse error. If the LLM
        returns garbage JSON, we fall back to a single subtask that's
        just the original query. This means the pipeline always produces
        SOMETHING, even if it's not optimal. The user gets a result
        instead of an error screen.
        
        This is the "graceful degradation" pattern — prefer a degraded
        result over a complete failure.
    """
    
    # ── Step 1: Read input from state ──
    query = state["query"]
    
    # ── Step 2: Emit "starting" event ──
    # We build a list of log events and return them all at once.
    # Each event gets appended to state["agent_logs"] via the reducer.
    logs = [
        make_event(
            agent="orchestrator",
            status="running",
            message=f"Analyzing query: '{query}'",
        )
    ]
    
    # ── Step 3: Initialize the LLM ──
    #
    # ChatGoogleGenerativeAI is LangChain's wrapper around the Gemini API.
    # 
    # WHY temperature=0.3?
    #   The Orchestrator needs to be CREATIVE enough to think of different
    #   research angles (so not 0.0), but RELIABLE enough to produce valid
    #   JSON consistently (so not 0.9). 0.3 is a sweet spot for structured
    #   creative tasks.
    #
    # WHY INITIALIZE INSIDE THE FUNCTION?
    #   We could make the LLM a module-level variable, but that would:
    #   - Run at import time (breaks tests that don't have API keys)
    #   - Make it impossible to use different settings per call
    #   Creating it fresh each call is cheap (it's just object construction,
    #   no API call happens until .invoke()).
    
    llm = ChatGoogleGenerativeAI(
        model=settings.LLM_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0.3,
    )
    
    # ── Step 4: Call the LLM ──
    #
    # .invoke() sends the prompt to the Gemini API and blocks until
    # the full response is received. For streaming, you'd use .stream(),
    # but since we need the complete JSON to parse it, streaming doesn't
    # help here.
    #
    # The response object has a .content attribute (str) with the text.
    
    response = llm.invoke(ORCHESTRATOR_PROMPT.format(query=query))
    
    # ── Step 5: Parse the JSON response ──
    #
    # This is where things can go wrong. LLMs are probabilistic —
    # even with explicit instructions, they sometimes:
    #   - Add preamble text before the JSON
    #   - Wrap JSON in markdown code fences
    #   - Return invalid JSON (missing comma, trailing comma, etc.)
    #   - Return a JSON object instead of an array
    #
    # Our try/except handles ALL of these gracefully.
    
    try:
        plan = _parse_llm_json(response.content)
        
        # VALIDATION: Ensure we got a list of dicts with required keys
        # This catches cases where the JSON is valid but wrong shape
        # (e.g., the LLM returns a single object instead of an array)
        if not isinstance(plan, list) or len(plan) == 0:
            raise ValueError("Expected a non-empty JSON array")
        
        for task in plan:
            if not all(key in task for key in ("id", "description", "search_query")):
                raise ValueError(f"Subtask missing required keys: {task}")
        
        logs.append(
            make_event(
                agent="orchestrator",
                status="completed",
                message=f"Created {len(plan)} research subtasks",
                data={"plan": plan},
            )
        )
        
    except (json.JSONDecodeError, ValueError, IndexError) as e:
        # ── FALLBACK: Single-task plan ──
        # Rather than crash, we create a minimal plan using the original
        # query. The pipeline will still work — it just won't benefit
        # from query decomposition.
        
        logs.append(
            make_event(
                agent="orchestrator",
                status="error",
                message=f"Failed to parse plan: {e}. Using fallback.",
            )
        )
        
        plan = [
            {
                "id": 1,
                "description": query,
                "search_query": query,
            }
        ]
        
        logs.append(
            make_event(
                agent="orchestrator",
                status="completed",
                message="Fallback: Created 1 subtask from original query",
                data={"plan": plan},
            )
        )
    
    # ── Step 6: Return partial state update ──
    #
    # We only return the fields we modified: plan and agent_logs.
    # LangGraph will:
    #   - SET state["plan"] = plan  (overwrite, no reducer)
    #   - APPEND our logs to state["agent_logs"]  (via append_list reducer)
    #   - Leave all other fields (query, draft, etc.) unchanged
    
    return {
        "plan": plan,
        "agent_logs": logs,
    }