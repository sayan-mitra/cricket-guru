"""Central config. Everything env-overridable so the same code runs locally
(Postgres socket + on-disk Qdrant) or in Docker (service URLs)."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]      # the project root
DATA_DIR = ROOT / "data"

# Load keys/settings from a .env (gitignored) if present.
load_dotenv(ROOT / ".env")
load_dotenv(ROOT / "backend" / ".env")

# --- Postgres ---
PG = dict(
    dbname=os.environ.get("CG_DB", "cricket_guru"),
    user=os.environ.get("CG_USER", "sayanmitra"),
    host=os.environ.get("CG_HOST", "/tmp"),      # Postgres.app socket dir
    port=int(os.environ.get("CG_PORT", "5432")),
    password=os.environ.get("CG_PASSWORD"),        # None locally (socket auth)
)

# --- Vector store ---
# CG_QDRANT_URL set (e.g. in Docker) -> talk to a Qdrant server; else on-disk.
QDRANT_URL = os.environ.get("CG_QDRANT_URL")
QDRANT_PATH = str(DATA_DIR / "qdrant")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384                                   # bge-small output dimension


def collection(source: str, chunking: str) -> str:
    """Qdrant collection name — one per (source, chunking variant). Sources:
    wiki (encyclopedic prose) and rules (authoritative rule books)."""
    return f"{source}_{chunking}"


def wiki_collection(chunking: str) -> str:
    return collection("wiki", chunking)


# --- Models & keys (from .env) ---
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# Web-search backends (freshness tool). Tavily/Brave are keyed APIs (robust under
# load); without a key it falls back to scraping DuckDuckGo (works, but throttles).
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY")
# Provider-qualified model strings (Pydantic AI format "<provider>:<model>").
# All-Anthropic: Sonnet answers, Haiku cross-judges. A non-Anthropic judge would make
# the same-vs-cross bias check stronger (see docs/how-it-works.md), but keeps it OpenAI-free.
ANSWERER_MODEL = os.environ.get("CG_ANSWERER_MODEL", "anthropic:claude-sonnet-5")
# Fast, cheap model for the mechanical legs — the input gate (classify) and the stats phraser (rows ->
# sentence). Pure classify/format work that doesn't need the answerer's reasoning, so Haiku ~halves
# their latency. NOT the groundedness guard: that's a judgment call and Haiku over-rejects (see
# guardrails._ground_agent).
FAST_MODEL = os.environ.get("CG_FAST_MODEL", "anthropic:claude-haiku-4-5")
JUDGE_CROSS_MODEL = os.environ.get("CG_JUDGE_MODEL", "anthropic:claude-haiku-4-5")
# Serving critic (CRAG groundedness gate). Its own knob so it can run on a strong
# model (Opus) independent of the answerer and the eval judge. Falls back to the
# answerer until the Anthropic key lands.
CRITIC_MODEL = os.environ.get("CG_CRITIC_MODEL", ANSWERER_MODEL)
# Below this max retrieval cosine, the critic calls it a retrieval_gap. A
# conservative floor, not a separator: calibration showed answerable gold scores
# 0.65+ while off-corpus probes still score 0.62-0.78, so cosine can't cleanly
# tell them apart. 0.55 sits below the answerable band, so it only fires when
# retrieval genuinely whiffs. The coverage flag + groundedness do the real work.
CRITIC_THRESHOLD = float(os.environ.get("CG_CRITIC_THRESHOLD", "0.55"))

# Per-request deadline on every LLM call. Without one the client blocks forever on a socket the far
# end has already dropped — an eval run sat on a dead CLOSE_WAIT connection for 80 minutes, and the
# same hang in serving would freeze the app with no error to show the user.
LLM_TIMEOUT = float(os.environ.get("CG_LLM_TIMEOUT", "120"))


def have_llm_keys() -> bool:
    return bool(ANTHROPIC_API_KEY or OPENAI_API_KEY)


# --- Pipeline mode: which variant of each leg to run ---
# Baseline is the simplest variant of every leg; experiments flip one at a time.
@dataclass(frozen=True)
class PipelineConfig:
    chunking: str = "fixed"      # fixed | structural       (L1)
    retrieval: str = "dense"     # dense | hybrid           (L3)
    router: str = "rule"         # rule | agent             (L5)
    judge: str = "same"          # same | cross             (L6)


BASELINE = PipelineConfig()
