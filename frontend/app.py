"""Cricket Guru — Streamlit frontend.

Chat-first. The workspace is a conversation (Ask), with a Chat/Compare toggle at the top;
a left nav rail switches to Traces and the How-it-works reference.

  Ask · Chat     — Mode A serving: one path, the critic decides ship / web / abstain.
  Ask · Compare  — Mode B fan-out: the same engine across lanes, display-only swimlane.

    PYTHONPATH=backend streamlit run frontend/app.py
"""
import json
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from cricket_guru.config import DATA_DIR          # noqa: E402
from cricket_guru.serve import compare, serving_engine   # noqa: E402

st.set_page_config(page_title="Cricket Guru", page_icon="🏏", layout="wide",
                   initial_sidebar_state="expanded")

# Chat-native shell: drop Streamlit's chrome, centre the column, warm the accents.
# Keep the sidebar collapse/expand control visible so a collapsed nav can always be reopened.
st.markdown("""
<style>
  /* hide only the chrome actions (Deploy/menu/status) + rainbow bar — NOT the whole toolbar,
     which also holds the sidebar expand button */
  [data-testid="stToolbarActions"], [data-testid="stDecoration"], footer {visibility: hidden;}
  [data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapseButton"] {
    visibility: visible !important; opacity: 1 !important; z-index: 9999 !important;}
  .block-container {padding-top: 2.5rem; padding-bottom: 6rem; max-width: 1040px;}
  section[data-testid="stSidebar"] {border-right: 1px solid #E6E6DF;}
  [data-testid="stChatInput"] {border-color: #1F7A5A33;}
</style>
""", unsafe_allow_html=True)

CHAT_CAP = 5
CHAT_FILE = DATA_DIR / "session" / "chat.json"   # survives browser refresh (session_state does not)


def _load_chat():
    try:
        return json.loads(CHAT_FILE.read_text()) if CHAT_FILE.exists() else []
    except Exception:
        return []


def _save_chat(chat):
    CHAT_FILE.parent.mkdir(parents=True, exist_ok=True)
    # cosine scores can be numpy floats — coerce anything json can't take natively
    CHAT_FILE.write_text(json.dumps(
        chat, default=lambda o: float(o) if hasattr(o, "__float__") else str(o)))

# Sample questions, each chosen to show a different critic outcome.
SAMPLES = [
    ("Who scored the most runs in IPL 2016?", "stats-SQL → ships a computed number"),
    ("Is there a free hit after a no-ball?", "rule books → ships an authoritative answer"),
    ("Who has taken the most Test wickets ever?", "all-time record outside the data window → abstains rather than guess"),
    ("Why was the 2019 World Cup final so controversial?", "narrative prose from the encyclopedia"),
    ("What is a doosra?", "corpus can't answer it → abstains rather than guess"),
]

BADGE = {
    "ok": ":green[**✓ shipped**]",
    "retrieval_gap": ":orange[**🌐 web fallback**]",
    "hallucination": ":red[**⚠ abstained**]",
    "blocked": ":gray[**⛔ blocked**]",
}

NAV = {"Ask": "💬  Ask", "Traces": "🔎  Traces", "How": "📖  How it works"}
MODES = ["💬 Chat", "🔀 Compare"]

# --- true mermaid via mermaid.js (components.html is not under the artifact CSP, so the CDN
#     import resolves; the local app has network). Theme tuned to the light/green canvas.
#     Diagrams in inactive st.tabs start display:none, so mermaid's startOnLoad would render into a
#     zero-width box and never redraw. Instead we poll until the container has width (its tab is
#     shown), then render once with mermaid.render(). ---
def mermaid(code, height=480):
    src = json.dumps(code)
    components.html(f"""
      <div id="mm"></div>
      <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{startOnLoad:false, theme:'base', themeVariables:{{
          primaryColor:'#EAF3EE', primaryBorderColor:'#1F7A5A', primaryTextColor:'#1A1A1A',
          lineColor:'#1F7A5A', secondaryColor:'#F1F2ED', fontFamily:'sans-serif', fontSize:'14px'}}}});
        const src = {src}, el = document.getElementById('mm');
        let done = false;
        async function draw() {{
          if (done || document.body.clientWidth === 0) return;   // still hidden — wait
          done = true;
          try {{
            const r = await mermaid.render('g' + Math.random().toString(36).slice(2), src);
            el.innerHTML = r.svg;
          }} catch (e) {{ done = false; el.innerHTML = '<pre style="color:#b00020">' + (e.message||e) + '</pre>'; }}
        }}
        const iv = setInterval(() => {{ if (done) clearInterval(iv); else draw(); }}, 250);
        draw();
      </script>""", height=height, scrolling=True)

ARCH_DIAGRAM = """
flowchart TB
  Q([cricket question]) --> G[guardrails: cricket-gate · injection · safety]
  G --> RT{router}
  RT -->|ReAct agent · or keyword rule-router| ARMS{{pick an answer arm}}
  ARMS --> S[stats-SQL<br/>Cricsheet Postgres]
  ARMS --> WK[text-RAG · wiki<br/>paragraph chunks · hybrid]
  ARMS --> RB[text-RAG · rules<br/>clause chunks · dense]
  ARMS --> WB[web search<br/>Tavily → Brave → DuckDuckGo]
  S --> C[answer critic · CRAG]
  WK --> C
  RB --> C
  WB --> C
  C -->|ok| SHIP([ship ✓])
  C -->|retrieval_gap| WEB([web + caveat 🌐])
  C -->|hallucination| ABS([abstain ⚠])
"""

REACT_DIAGRAM = """
flowchart LR
  Q([question]) --> T{which tool?}
  T -->|stats| S[cricket_stats]
  T -->|rules or prose| K[semantic_search]
  T -->|out of corpus| W[web_search]
  S --> O[observe result]
  K --> O
  W --> O
  O --> D{enough to answer?}
  D -->|no, loop back| T
  D -->|yes| A[compose answer]
  A --> OG[output guardrail:<br/>groundedness]
  OG --> OUT([answer])
  classDef hot fill:#EAF3EE,stroke:#1F7A5A,stroke-width:2px;
  class T,O,D hot
"""

CRAG_DIAGRAM = """
flowchart TB
  A[answer + retrieval evidence] --> Q1{coverage flag or low score?}
  Q1 -->|yes · cheap gate| RG[retrieval_gap<br/>→ web + caveat 🌐]
  Q1 -->|no| Q2{groundedness ok?}
  Q2 -->|grounded| OK[ok → ship ✓]
  Q2 -->|core unsupported| HL[hallucination<br/>→ abstain ⚠]
"""


# --- trace rendering -------------------------------------------------------

def _io(s):
    """Render a step's input/output prompts, if present. A retrieve step carries structured `hits`
    (cosine · title · snippet) — show those as a table instead of the flat output string."""
    if s.get("input"):
        st.caption("input")
        st.code(str(s["input"])[:8000], language=None)
    if s.get("hits"):
        st.caption("retrieved chunks")
        st.dataframe([{"cosine": h["cosine"], "title": h["title"], "snippet": h["snippet"]}
                      for h in s["hits"]], hide_index=True, use_container_width=True)
    elif s.get("output"):
        st.caption("output")
        st.code(str(s["output"])[:8000], language=None)


def _render_node(s, kids, depth=0):
    """Render one span (expander with I/O + sub-steps), then its children indented under it."""
    label = f"{'↳ ' if depth else ''}{s['kind']} · {s['name']} · {s.get('ms', 0)} ms"
    col = st.columns([0.04, 0.96])[1] if depth else st.container()   # indent children
    with col:
        if s.get("input") or s.get("output") or s.get("steps"):
            with st.expander(label):
                _io(s)
                for sub in s.get("steps", []):
                    st.markdown(f"**↳ {sub.get('name', '')}**")
                    _io(sub)
        else:
            st.text(label)
    for c in kids.get(s["name"], []):
        _render_node(c, kids, depth + 1)


def render_spans(spans):
    """Render the trace as a call tree: top-level legs in sequence (↓), each leg's callees
    (tools under the agent, fallbacks under the critic) indented beneath it via parent links.
    Must NOT be called inside an expander — Streamlit forbids nesting them."""
    kids = {}
    for s in spans:
        if s.get("parent") is not None:
            kids.setdefault(s["parent"], []).append(s)
    roots = [s for s in spans if s.get("parent") is None]
    for i, s in enumerate(roots):
        _render_node(s, kids)
        if i < len(roots) - 1:
            st.markdown("↓")


# --- serving ---------------------------------------------------------------

def chat_serve(question, history):
    """Mode A serving with conversation history. Not cached — each turn is context-unique."""
    r = serving_engine().serve(question, history)
    b = r.base
    return {"text": r.text, "verdict": r.verdict, "reason": r.reason,
            "latency_ms": r.latency_ms, "tokens": b.tokens, "grounded": b.grounded,
            "retrieval_score": b.retrieval_score, "tools": b.tool_trace,
            "sources": b.sources, "trace": b.trace}


@st.cache_data(show_spinner=False)
def compare_cached(question, axis):
    return compare(question, axis)


# --- Ask: chat (Mode A) ----------------------------------------------------

def _new_chat():
    st.session_state.chat = []
    _save_chat([])
    for k in [k for k in list(st.session_state) if k.startswith("tr_")]:
        del st.session_state[k]


def render_chat():
    st.session_state.setdefault("chat", _load_chat())   # reload persisted chat after a refresh
    chat = st.session_state.chat

    if chat:                                            # a chat exists — offer New chat at any point
        st.columns([0.72, 0.28])[1].button(
            "🔄 New chat", on_click=_new_chat, use_container_width=True)

    for i, turn in enumerate(chat):
        with st.chat_message("user"):
            st.markdown(turn["q"])
        with st.chat_message("assistant"):
            a = turn["a"]
            st.markdown(a["text"])
            st.caption(f"{BADGE.get(a['verdict'], a['verdict'])}  ·  {a['reason'][:120]}  ·  "
                       f"{a['latency_ms']/1000:.1f}s · {a['tokens']} tok · " + " → ".join(a["tools"]))
            if a["trace"] and st.toggle("step-by-step trace", key=f"tr_{i}"):
                render_spans(a["trace"])

    n = len(chat)
    pending = st.session_state.pop("pending", None)
    if n >= CHAT_CAP:
        st.info(f"Reached {CHAT_CAP} questions — tap New chat above to keep going.")
        return
    typed = st.chat_input(f"Ask a cricket question…   ({n}/{CHAT_CAP})")
    q = pending or typed
    if q:
        history = [(t["q"], t["a"]["text"]) for t in chat]
        with st.spinner("Routing, answering, checking…"):
            a = chat_serve(q, history)
        chat.append({"q": q, "a": a})
        _save_chat(chat)
        st.rerun()


# --- Ask: compare (Mode B) -------------------------------------------------

def render_compare():
    st.caption("The same engine, fanned out over one axis at a time. Display-only — "
               "every lane's answer and critic verdict, no winner picked.")
    axis = st.radio("Compare", ["router", "chunking"], horizontal=True,
                    format_func=lambda x: {"router": "Routing (rule vs agent)",
                                           "chunking": "Chunking (fixed vs structural)"}[x])
    st.caption({"router": "Full system; chunking + retrieval held at baseline.",
                "chunking": "Text-RAG arm only, no router; retrieval held at baseline."}[axis])
    q = st.text_input("Question to compare", key="cmp_q",
                      placeholder="Who scored the most runs in IPL 2016?")
    if not q:
        st.info("Type a question above to fan it out across the lanes.")
        return
    if not st.button("Run comparison", type="primary"):
        return
    with st.spinner(f"Running {axis} lanes…"):
        r = compare_cached(q, axis)
    lanes = r["lanes"]
    cols = st.columns(len(lanes) + 1)
    for col, L in zip(cols, lanes):
        with col, st.container(border=True):
            st.markdown(f"**{L['label']}**")
            st.markdown(BADGE.get(L["verdict"], L["verdict"]))
            bits = [f"{L['latency_ms']/1000:.1f}s"]
            if L["retrieval_score"] is not None:
                bits.append(f"match {L['retrieval_score']:.2f}")
            if L["tokens"]:
                bits.append(f"{L['tokens']} tok")
            st.caption(" · ".join(bits))
            st.markdown(L["text"][:600])
            st.caption("Tools: " + " → ".join(L["tools"]))
            if L["reason"]:
                st.caption(f"critic: {L['reason'][:120]}")
    # Gold lane — greyed when no semantic match, present but honest.
    with cols[-1], st.container(border=True):
        g = r["gold"]
        if g:
            st.markdown("**gold reference** ✓")
            st.caption(f"matched {g['id']} · sim {g['score']:.2f}")
            st.markdown(g["reference"][:600])
        else:
            st.markdown(":gray[**gold reference**]")
            st.caption("no gold hit — this question isn't in the curated set")


def render_ask():
    st.markdown("#### Ask about stats, rules, or cricket history")
    st.caption("It routes each question to the right source — the Cricsheet database, the rule "
               "books, the encyclopedia, or the web. A critic checks the answer before it ships, "
               "and every step is on the record.")
    # Keyed widget + callback-set state, so a sample button can force back to Chat
    # (passing `default` alone won't override the widget's remembered selection).
    st.session_state.setdefault("mode_sel", MODES[0])
    mode = st.segmented_control("mode", MODES, key="mode_sel", label_visibility="collapsed")
    (render_compare if mode == "🔀 Compare" else render_chat)()


# --- tools (nav rail) ------------------------------------------------------

def render_traces():
    st.subheader("Traces")
    st.caption("Every run's timed spans (LLM / tool / SQL / guardrail calls), newest first.")
    path = DATA_DIR / "traces" / "traces.jsonl"
    if not path.exists():
        st.info("No traces yet — ask a question first.")
        return
    runs = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    recent = list(reversed(runs[-25:]))
    labels = [f"{r['question'][:60]} · {r['total_ms']} ms" + (" ⛔" if r.get("blocked") else "")
              for r in recent]
    i = st.selectbox("Pick a run", range(len(recent)), format_func=lambda i: labels[i])
    run = recent[i]
    st.markdown(f"**{run['answer'][:400]}**")
    render_spans(run["spans"])


# Per-leg experiment notes, keyed to experiments.json. Order = how a query flows through the system.
EXP_ORDER = ["L5_routing", "L3_retrieval", "L1_chunking", "rules_chunking", "rules_arm", "L6_judge"]
EXP_NOTES = {
    "L5_routing": ("1 · Routing (full system)", "Where the query enters. LLM tool-calling agent vs a "
                   "keyword rule-router — the agent picks the arm."),
    "L3_retrieval": ("2 · Retrieval · wiki", "The chosen arm fetches chunks. Hybrid (dense + BM25) vs "
                     "dense-only."),
    "L1_chunking": ("3 · Chunking · wiki", "What retrieval searches over. Structural (paragraph) vs "
                    "fixed windows — within noise on prose."),
    "rules_chunking": ("3 · Chunking · rules", "Same leg, rules corpus. Clause-split vs fixed — clause-"
                       "aware is qualitatively right; the reward is within noise."),
    "rules_arm": ("Rules arm", "The rules RAG arm on its dense-retrieval baseline."),
    "L6_judge": ("4 · Judge — same vs cross", "NOT the judge's accuracy. The number is the AGENT's "
                 "narrative accuracy as scored by each judge (gpt grading its own answers vs Sonnet "
                 "grading them); the GAP is the self-preference check — cross ≥ same means no same-judge "
                 "inflation. The judges themselves are ~95–100% human-aligned (Phase-B)."),
}


@st.cache_data(show_spinner=False)
def _chunk_demo():
    """Real before/after on the rules corpus: fixed windows vs clause-aware, on the Hit Wicket law.
    Target shrunk to ~320 chars so the boundaries fit on screen — the mechanism is the point."""
    try:
        from cricket_guru.index.chunking import chunk_fixed, chunk_structural
        docs = json.loads((DATA_DIR / "rules" / "rules.json").read_text())
        doc = next(d for d in docs if "35.2" in d["text"] and "Hit wicket" in d["text"]
                   and len(d["text"]) > 900)
        raw = doc["text"]
        fixed = [" ".join(c.split()) for c in chunk_fixed(raw, target=320, overlap=40)][:4]
        clause = [" ".join(c.split()) for c in chunk_structural(raw, target=320)][:4]
        return doc["title"].strip(), " ".join(raw.split())[:200], fixed, clause
    except Exception:
        return None

# Gold sets: (file, label, count-fallback, scoring one-liner).
GOLD_CARDS = [
    ("stats_gold", "stats", "Exact-match against the SQL oracle — the computed number or label must "
     "appear in the answer. Cricsheet is the objective truth for stats."),
    ("rules_gold_corpus", "rules", "Clause-grounded: each reference IS a rulebook clause, and a verify "
     "filter drops any question the clause can't actually answer (it caught a bad concussion-deadline "
     "item). Scored by LLM-judge."),
    ("narrative_gold_corpus", "narrative", "Corpus-grounded encyclopedia prose: each reference IS a "
     "Wikipedia passage, with a verify filter dropping questions the passage can't answer — the same "
     "rebuild as rules. Scored by LLM-judge, lenient on phrasing, strict on facts."),
    ("multistep_gold", "multistep", "Verifies the ReAct loop: the judged answer AND the trace must show "
     "2+ tool types composed. A right number reached in one hop still fails — it measures decomposition."),
]


def render_howitworks():
    st.markdown("#### How Cricket Guru works")
    st.caption("The architecture, the ReAct loop, the CRAG critic, and what each leg taught us — "
               "plus the experiments and the gold sets behind the numbers.")
    tabs = st.tabs(["Architecture", "ReAct loop", "CRAG critic", "Learnings", "Experiments", "Gold set"])

    with tabs[0]:
        st.markdown("One question, routed to the right source, checked before it ships.")
        mermaid(ARCH_DIAGRAM, height=560)

    with tabs[1]:
        st.markdown("The agent **thinks, calls a tool, observes, and loops** until it can answer — "
                    "the highlighted `think → observe → decide` cycle is the ReAct loop. "
                    "Multistep gold exists to verify this decomposition actually happens.")
        mermaid(REACT_DIAGRAM, height=440)

    with tabs[2]:
        st.markdown("**Corrective RAG** on the serving path: a critic reads every answer and its "
                    "evidence, then decides ship / web / abstain.")
        mermaid(CRAG_DIAGRAM, height=420)
        st.markdown(
            "- **ok** → ship the answer.\n"
            "- **retrieval_gap** → the corpus genuinely can't reach it (an all-time record outside the "
            "2001+ window, or every chunk below τ). The web has what we lack — ship the web answer with "
            "a *'verified from the web, not our corpus'* caveat.\n"
            "- **hallucination** → retrieval was fine but the answer overreaches. The web won't fix "
            "that, so **abstain** (\"I can't verify this\") instead of swapping one guess for another.\n\n"
            "The cheap gate (coverage flag / score < τ=0.55) is deterministic and free; only answers "
            "that clear it pay for the groundedness call on its own `CG_CRITIC_MODEL`.")

    with tabs[3]:
        st.caption("What each leg taught us — with the real example that drove the change.")

        st.markdown("**Chunking** — rules split on clause numbers, wiki on paragraphs. Structural "
                    "chunking degrades to arbitrary windows on rulebook PDFs (no paragraph breaks, only "
                    "numbered clauses), so rules needed a clause-aware splitter; wiki kept paragraphs.")
        demo = _chunk_demo()
        if demo:
            title, head, fixed, clause = demo
            st.caption(f"Example — the Hit Wicket law (`{title}`), chunk size ~320 chars so boundaries fit:")
            st.markdown(f"> {head}…")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(":red[**Fixed windows**] — cut mid-clause")
                for c in fixed:
                    st.code(c[:170], language=None)
            with c2:
                st.markdown(":green[**Clause-aware**] — one rule per chunk")
                for c in clause:
                    st.code(c[:170], language=None)
        st.divider()

        st.markdown("**Retrieval** — dense for rules, hybrid for wiki. The BM25 half of hybrid matches "
                    "surface terms to the wrong clause, so the rules arm uses pure dense cosine.")
        st.caption("Example — query *“what happens when the ball hits a fielder's helmet on the ground?”*")
        r1, r2 = st.columns(2)
        with r1:
            st.markdown(":red[**Hybrid top hit**] — wrong (lexical ‘fielder’)")
            st.code("28.2.3  If a fielder illegally fields the ball…   (0.748)", language=None)
        with r2:
            st.markdown(":green[**Dense top hit**] — right (the helmet clause)")
            st.code("28.3.2  If the ball… strikes the protective helmet…   (0.779)", language=None)
        st.caption("recall@k confirms the split: dense leads @1 on rules, hybrid @1 on wiki (see Experiments). "
                   "A **cross-encoder reranker** on the wiki arm then lifts recall@1 60→80% — the right passage "
                   "is usually retrieved but ranked too low, and re-scoring the top-20 fixes it. Measured to "
                   "hurt rules (already ~90% @1), so it's wiki-only.")
        st.divider()

        st.markdown("**Routing** — a small answerer (gpt-mini) way-found to route on par with a strong "
                    "model via richer tool descriptions and coverage notes, not hardcoded keyword rules.")
        st.caption("Example — the same router discriminates by intent and by data coverage:")
        st.markdown("- *“most runs in IPL 2016?”* → **cricket_stats** (in-window SQL)\n"
                    "- *“why was the 2019 final controversial?”* → **semantic_search** (wiki prose)\n"
                    "- *“most Test wickets ever?”* → **web_search** (all-time record, outside the 2001+ data)")
        st.caption("The biggest narrative finding: it was **route-capped, not retrieval-capped**. The wiki "
                   "arm alone answers 96% of narrative questions, but the agent managed only 64% — it "
                   "misrouted history/record questions (which *look* statistical: 'most WC wickets', "
                   "'Kohli's captaincy record') to cricket_stats, where they can't be answered. Making the "
                   "tool contracts concrete about what each source can and can't hold lifted the agent to "
                   "~84%. The bottleneck was orchestration, not retrieval.")
        st.divider()

        st.markdown("**Stats-SQL** — the biggest lever was the **schema**. A bare column list made the model "
                    "infer column meaning from names, and it inferred wrong (`d.bowling_team`, which doesn't "
                    "exist; `runs_batter` treated as an innings total). It now gets an **annotated schema** — "
                    "every column's meaning, units, and granularity — plus a CTE-capable read-only guard, a "
                    "self-correcting retry loop (Postgres errors fed back), and ILIKE surname matching.")
        st.caption("Example — *“which match did Kohli score his 82 in?”* — 82 is an innings total, but "
                   "`runs_batter` is runs off one ball (0–6):")
        s1, s2 = st.columns(2)
        with s1:
            st.markdown(":red[**Bare column list**] — 82 read as a per-ball value")
            st.code("GROUP BY … runs_batter\nHAVING SUM(…) = 82\n→ []  (no ball scores 82)", language=None)
        with s2:
            st.markdown(":green[**Annotated schema**] — 82 is a SUM")
            st.code("SUM(runs_batter)\nGROUP BY match, innings = 82\n→ match on 2022-10-23 ✓", language=None)
        st.caption("Same class as `d.bowling_team` — both fixed by telling the model what each column means, "
                   "not letting it guess from the name. Like a database MCP would, but authored once.")
        st.caption("Two more robustness fixes: the schema now states what the DB **can't** hold — no "
                   "captaincy (there is no captain column), no pre-window records — so those route to prose "
                   "instead of the model fabricating them from lineups; and an **empty-result retry** "
                   "loosens over-constrained queries (the model kept over-filtering 'runs off 35 balls in "
                   "the 2025 final' down to nothing).")
        st.divider()

        st.markdown("**Critic (CRAG)** — grades every finished answer: **ok** → ship, **retrieval_gap** → "
                    "web-verify, **hallucination** → abstain. The coverage call is the critic model reasoning "
                    "about the data window, not a regex — a T20I record is complete (T20Is only exist from "
                    "2005), an all-time Test record isn't (Tests date to 1877).")
        st.caption("Example — *“highest India–Australia T20 total?”* → **ships 235** (in-window, complete); "
                   "the old regex wrongly sent it to the web and got 272. *“most Test wickets ever?”* reaches "
                   "before the window → web-verifies (Muralitharan 800, with sources) or **abstains with the "
                   "reason** — the web has handed back wrong figures here (272, Warne 708), so a grounded-or-"
                   "honest answer beats a confident wrong one.")
        st.divider()

        st.markdown("**Judge** — cross-vendor (gpt answers, Sonnet judges) to dodge same-model self-"
                    "preference. Validated against human labels — no same-judge bias showed up.")
        st.caption("Example — on 21 human-labeled items the same/gpt judge agreed 21/21 and the cross/"
                   "Sonnet judge 20/21; the one miss was Sonnet passing an answer that addressed only half "
                   "the question, which gpt caught. So no evidence of same-judge bias on this set.")

    with tabs[4]:
        path = DATA_DIR / "results" / "experiments.json"
        if not path.exists():
            st.info("No results yet. Run: `python -m cricket_guru.eval.run_experiments`")
        else:
            res = json.loads(path.read_text())
            st.markdown(
                f"**What each percentage means.** Every variant is a full pipeline run over a fixed gold "
                f"set (~{res.get('sample_size')} questions/leg). The number is end-to-end **answer "
                f"accuracy** — the share the whole system got right with just that one leg swapped to the "
                f"named approach, everything else held at baseline. Read each leg as *baseline vs "
                f"advanced*: the gap is what the technique bought.")
            st.caption("Ordered the way a query flows: route in, retrieve, over chunks, then judge the answer.")
            st.caption("The narrative legs (chunking · retrieval · judge) are scored on the **corpus-grounded** "
                       "gold — reference = the actual Wikipedia passage — a stricter bar than the old self-"
                       "written references, which is why the judge leg reads lower (56–60%) but truer.")
            st.caption("Diagnostic finding beyond the ablation: narrative was **route-capped, not retrieval-"
                       "capped**. The wiki arm alone answers 96%, but the agent misrouted history/record "
                       "questions to stats and managed 64%; sharpening the tool contracts lifted it to ~84%. "
                       "So the L6 judge numbers here (base retrieval, no routing fix) understate the fixed system.")
            for key in EXP_ORDER:
                if key not in res or not isinstance(res[key], dict):
                    continue
                variants = res[key]
                title, note = EXP_NOTES.get(key, (key, ""))
                with st.container(border=True):
                    st.markdown(f"**{title}**")
                    st.caption(note)
                    for col, (variant, score) in zip(st.columns(len(variants)), variants.items()):
                        col.metric(variant, f"{score:.0%}")

            rpath = DATA_DIR / "results" / "recall.json"
            if rpath.exists():
                rc = json.loads(rpath.read_text())
                st.divider()
                st.markdown(f"**Retrieval recall@k**  ·  {rc['mode']} match — a direct retrieval metric, "
                            f"no LLM or judge")
                st.caption("Ask the question, take the top-k chunks — was the gold passage among them? "
                           "`+rr` = with the cross-encoder reranker. It's a big win on wiki "
                           "(structural/hybrid @1 60→80%), so the **serving wiki arm uses it**; on rules it "
                           "slightly hurts @1 (already ~90%), so rules stays on plain dense.")
                for s in rc.get("sources", []):
                    st.markdown(f"_{s['source']} corpus · n={s['n']}_")
                    rows = [{"chunking": g["chunking"], "retrieval": g["retrieval"],
                             "@1": f"{g['recall']['1']:.0%}", "@3": f"{g['recall']['3']:.0%}",
                             "@5": f"{g['recall']['5']:.0%}",
                             "@1 +rr": f"{g['reranked']['1']:.0%}" if g.get("reranked") else "—",
                             "@3 +rr": f"{g['reranked']['3']:.0%}" if g.get("reranked") else "—",
                             "@5 +rr": f"{g['reranked']['5']:.0%}" if g.get("reranked") else "—"}
                            for g in s["grid"]]
                    st.dataframe(rows, hide_index=True, use_container_width=True)

    with tabs[5]:
        st.caption("Three question types plus a composition set — each scored the way that type demands. "
                   "Expand a set to read real items.")
        for file, label, scoring in GOLD_CARDS:
            p = DATA_DIR / "gold" / f"{file}.json"
            items = json.loads(p.read_text()) if p.exists() else []
            with st.expander(f"{label}  ·  {len(items)} items"):
                st.caption(scoring)
                for it in items[:3]:
                    st.markdown(f"**Q.** {it['question']}")
                    ref = it.get("reference") or it.get("answer") or it.get("answer_label")
                    st.markdown(f"_ref:_ {str(ref)[:400]}")
                    st.divider()


# --- shell: brand + nav rail, then the workspace ---------------------------

with st.sidebar:
    st.markdown("### 🏏 Cricket Guru")
    st.caption("Routes each question, checks the answer, shows its work.")
    nav = st.radio("Go to", list(NAV), format_func=lambda k: NAV[k], label_visibility="collapsed")
    if nav == "Ask":
        st.divider()
        st.caption("Try one — tap to ask:")
        for sample, why in SAMPLES:
            st.button(sample, key=f"s_{sample[:20]}", use_container_width=True, help=why,
                      on_click=lambda s=sample: st.session_state.update(pending=s, mode_sel="💬 Chat"))

{"Ask": render_ask, "Traces": render_traces, "How": render_howitworks}[nav]()
