"""
config.py — Centralized configuration for AgentForge.

WHY THIS FILE EXISTS:
    Every module in the app needs access to API keys and settings.
    Instead of scattering os.getenv() calls across 10 files, we
    centralize everything here. This gives us:
    
    1. One place to see ALL settings at a glance
    2. One place to add validation ("is the API key actually set?")
    3. Easy testing — mock this one module to test with fake keys
    4. No circular dependencies — this file imports nothing from our app

USAGE:
    from app.config import settings
    
    llm = ChatGoogleGenerativeAI(
        model=settings.LLM_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
    )
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

# load_dotenv() reads the .env file in the project root and injects
# its key=value pairs into os.environ. This happens once at import time.
# If .env doesn't exist, this is a no-op — no crash.
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """
    Application settings loaded from environment variables.
    
    WHY A DATACLASS?
    - frozen=True makes it immutable — settings can't be accidentally
      modified at runtime. This prevents a whole class of bugs where
      one module changes a setting and breaks another.
    - Type hints serve as documentation for what each setting is.
    - Default values make the app runnable even with minimal config.
    
    WHY NOT PYDANTIC BaseSettings?
    - Pydantic BaseSettings is great for larger apps, but it adds
      complexity (validators, .env parsing built-in, etc.) that we
      don't need yet. A frozen dataclass gives us 80% of the benefit
      with 20% of the complexity. We can upgrade later if needed.
    """
    
    # --- LLM Configuration ---
    GOOGLE_API_KEY: str = ""
    LLM_MODEL: str = "gemini-2.0-flash"
    
    # --- Search Tool Configuration ---
    TAVILY_API_KEY: str = ""
    
    # --- Agent Behavior ---
    # Maximum number of Writer revisions before accepting the draft
    MAX_REVISIONS: int = 2
    
    # Minimum Critic score (1-10) to accept a draft without revision.
    # Below this threshold, the draft gets sent back to the Writer.
    QUALITY_THRESHOLD: int = 7
    
 
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    def validate(self) -> list[str]:
        """
        Check that required settings are present.
        
        Returns a list of error messages. Empty list = all good.
        
        WHY A METHOD INSTEAD OF __post_init__?
        - We want the app to load even with missing keys (for testing,
          for showing help text, etc.). Crashing at import time is
          aggressive. Instead, we validate explicitly when we're about
          to do real work (e.g., right before running the pipeline).
        """
        errors = []
        
        if not self.GOOGLE_API_KEY or self.GOOGLE_API_KEY == "your_google_api_key_here":
            errors.append(
                "GOOGLE_API_KEY is not set. "
            )
        
        if not self.TAVILY_API_KEY or self.TAVILY_API_KEY == "your_tavily_api_key_here":
            errors.append(
                "TAVILY_API_KEY is not set. "
            )
        
        return errors


def _load_settings() -> Settings:
    """
    Factory function that reads os.environ and creates a Settings instance.
    
    WHY A FACTORY FUNCTION?
    - Keeps the Settings class pure (no os.environ dependency in it).
    - Makes testing trivial: in tests, you construct Settings(GOOGLE_API_KEY="fake")
      directly, bypassing this function entirely.
    """
    return Settings(
        GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY", ""),
        LLM_MODEL=os.getenv("LLM_MODEL", "gemini-2.0-flash"),
        TAVILY_API_KEY=os.getenv("TAVILY_API_KEY", ""),
    )


# ─── Module-level singleton ────────────────────────────────
# This runs once when any module does: from app.config import settings
# Every module shares the same Settings instance.
settings = _load_settings()