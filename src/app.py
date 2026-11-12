"""Streamwise : interface Streamlit du RAG sur la documentation Streamlit."""

import base64
import html
import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# ----------------------------------------------------------------------------
# Identité (modifiez ces deux lignes pour renommer l'application / l'assistant)
# ----------------------------------------------------------------------------
APP_NAME = "Streamwise"
BOT_NAME = "Nova"

NO_ANSWER_MARK = "Je ne sais pas"
NO_ANSWER = "Je ne sais pas répondre à cette question avec la documentation fournie."
LLM_MODEL = "openai/gpt-oss-20b"  # surchargeable via model.llm_name dans config.yaml

SUGGESTIONS = [
    ("📊", "How do I create a dataframe in Streamlit?"),
    ("⚡", "How can I cache data with st.cache_data?"),
    ("🧠", "How do I use session state?"),
    ("📈", "How do I display a chart?"),
]

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)


def build_prompt(context: str, question: str) -> str:
    return f"""Tu es un assistant technique expert sur Streamlit. Réponds à la question en te basant **uniquement** sur le contexte fourni ci-dessous.
Si l'information n'est pas présente dans le contexte ou si la question est hors du sujet de Streamlit, réponds strictement : "{NO_ANSWER}" N'invente rien.

Contexte :
{context}

Question : {question}
Réponse :"""


# ----------------------------------------------------------------------------
# Illustrations (SVG intégrés : aucune image externe à télécharger)
# ----------------------------------------------------------------------------
def data_uri(mime: str, raw: bytes) -> str:
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


def svg_uri(svg: str) -> str:
    return data_uri("image/svg+xml", svg.encode("utf-8"))


def streamlit_logo_uri() -> str | None:
    """Logo officiel de Streamlit, lu depuis votre installation (aucun téléchargement)."""
    try:
        png = (Path(st.__file__).parent / "static" / "favicon.png").read_bytes()
        return data_uri("image/png", png)
    except Exception:
        return None


LOGO_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="9" fill="#FFFFFF"/><rect x="6" y="9" width="20" height="14" rx="6" fill="#0A2533"/><ellipse cx="12.5" cy="16" rx="2.2" ry="2.8" fill="#5EEAD4"/><ellipse cx="19.5" cy="16" rx="2.2" ry="2.8" fill="#5EEAD4"/></svg>"""


def build_hero_svg(logo_uri: str | None) -> str:
    """Robot + orbite. Les icônes sont dessinées en vectoriel (pas d'emoji)."""
    icon_style = 'stroke="#5EEAD4" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" fill="none"'
    if logo_uri:
        fourth = f'<image href="{logo_uri}" x="280" y="176" width="28" height="28"/>'
    else:  # éclair de secours si le logo est introuvable
        fourth = f'<g transform="translate(294 190)" {icon_style}><path d="M2 -10 L-6 2 H0 L-2 10 L6 -2 H0 Z"/></g>'
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 360 250" fill="none">
<circle cx="180" cy="150" r="118" stroke="#5EEAD4" stroke-opacity=".35" stroke-width="2" stroke-dasharray="2 7" stroke-linecap="round"/>
<circle cx="180" cy="150" r="82" stroke="#FFFFFF" stroke-opacity=".28" stroke-width="2" stroke-dasharray="2 7" stroke-linecap="round"/>
<rect x="63" y="63" width="38" height="38" rx="10" fill="#0A3B49" stroke="#5EEAD4" stroke-opacity=".5"/>
<g transform="translate(82 82)" {icon_style}><rect x="-7" y="-9" width="14" height="18" rx="2.5"/><path d="M-3 -3 H3 M-3 1 H3 M-3 5 H1"/></g>
<rect x="259" y="63" width="38" height="38" rx="10" fill="#0A3B49" stroke="#5EEAD4" stroke-opacity=".5"/>
<g transform="translate(278 82)" {icon_style}><circle cx="-1.5" cy="-1.5" r="6.5"/><path d="M3.5 3.5 L8.5 8.5"/></g>
<rect x="47" y="171" width="38" height="38" rx="10" fill="#0A3B49" stroke="#5EEAD4" stroke-opacity=".5"/>
<g transform="translate(66 190)" {icon_style}><rect x="-9" y="-8" width="18" height="13" rx="4"/><path d="M-3 5 L-5 10 L2 5"/></g>
<rect x="275" y="171" width="38" height="38" rx="10" fill="#0A3B49" stroke="#5EEAD4" stroke-opacity=".5"/>
{fourth}
<line x1="180" y1="105" x2="180" y2="93" stroke="#CFDCE4" stroke-width="4" stroke-linecap="round"/>
<circle cx="180" cy="88" r="6" fill="#5EEAD4"/>
<rect x="120" y="130" width="12" height="28" rx="6" fill="#CFDCE4"/>
<rect x="228" y="130" width="12" height="28" rx="6" fill="#CFDCE4"/>
<rect x="130" y="105" width="100" height="78" rx="32" fill="#EAF1F5"/>
<rect x="141" y="116" width="78" height="56" rx="22" fill="#0A2533"/>
<ellipse cx="165" cy="140" rx="8.5" ry="10.5" fill="#5EEAD4"/>
<ellipse cx="195" cy="140" rx="8.5" ry="10.5" fill="#5EEAD4"/>
<path d="M173 158 Q180 164 187 158" stroke="#5EEAD4" stroke-width="3" stroke-linecap="round"/>
<rect x="146" y="188" width="68" height="50" rx="24" fill="#EAF1F5"/>
<rect x="166" y="204" width="28" height="6" rx="3" fill="#5EEAD4" fill-opacity=".85"/>
</svg>"""


STREAMLIT_LOGO = streamlit_logo_uri()
LOGO_URI = svg_uri(LOGO_SVG)
HERO_ART = svg_uri(build_hero_svg(STREAMLIT_LOGO))


def html_block(markup: str) -> str:
    """Supprime indentation et lignes vides : évite que Markdown prenne le HTML pour du code."""
    return "\n".join(line.strip() for line in markup.strip().splitlines() if line.strip())


# ----------------------------------------------------------------------------
# Design (CSS) : fond dégradé + grille sur toute la page, polices système
# ----------------------------------------------------------------------------
STYLES = """
<style>
:root {
    --ink: #E8F5F3;
    --muted: #9FBDBD;
    --line: rgba(255, 255, 255, .12);
    --glass: rgba(255, 255, 255, .06);
    --accent: #2DD4BF;
    --accent-soft: rgba(45, 212, 191, .14);
}
html, body, .stApp, .stMarkdown, button, input, textarea {
    font-family: 'Segoe UI', 'Inter', system-ui, -apple-system, 'Helvetica Neue', Arial, sans-serif;
}
.stApp {
    color: var(--ink);
    background:
        linear-gradient(rgba(255, 255, 255, .045) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, .045) 1px, transparent 1px),
        radial-gradient(900px 420px at 50% -8%, rgba(45, 212, 191, .22), transparent 62%),
        linear-gradient(165deg, #06222F 0%, #0A3B49 55%, #0B5560 100%);
    background-size: 34px 34px, 34px 34px, auto, auto;
    background-attachment: fixed;
}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li { color: var(--ink); }
[data-testid="stCaptionContainer"] { color: var(--muted); }

#MainMenu, footer { visibility: hidden; }
[data-testid="stToolbar"], [data-testid="stDecoration"] { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }

.block-container,
[data-testid="stBottomBlockContainer"] {
    max-width: 880px;
    padding-left: 1rem;
    padding-right: 1rem;
}
.block-container { padding-top: 1.2rem; padding-bottom: 7rem; }
[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stBottomBlockContainer"] { background: transparent; }

/* ---------- Navigation ---------- */
.nav { display: flex; align-items: center; justify-content: space-between; }
.nav .brand-inline { display: flex; align-items: center; gap: .55rem; color: #fff; font-weight: 800; font-size: 1.05rem; }
.nav-logo { width: 32px; height: 32px; }
.status {
    display: inline-flex; align-items: center; gap: .45rem;
    padding: .32rem .8rem; border-radius: 999px;
    background: rgba(255, 255, 255, .08);
    border: 1px solid rgba(255, 255, 255, .18);
    color: #D7F5EF; font-size: .78rem; font-weight: 600;
}
.status i {
    width: 7px; height: 7px; border-radius: 50%;
    background: #34D399; box-shadow: 0 0 0 3px rgba(52, 211, 153, .25);
}
.topbar { padding: .2rem 0 1rem; border-bottom: 1px solid var(--line); margin-bottom: 1.1rem; }

/* ---------- Page d'accueil ---------- */
.hero-body { text-align: center; padding: 2rem 0 .4rem; }
.eyebrow {
    display: inline-flex; align-items: center; gap: .45rem;
    padding: .3rem .85rem; margin-bottom: 1.1rem;
    border-radius: 999px; border: 1px solid rgba(94, 234, 212, .4);
    background: rgba(94, 234, 212, .06);
    color: #99F6E4; font-size: .72rem; font-weight: 700;
    letter-spacing: .08em; text-transform: uppercase;
}
.eyebrow img { width: 18px; height: 18px; border-radius: 50%; }
.hero-title {
    color: #fff; font-size: clamp(1.9rem, 5vw, 2.7rem); font-weight: 800;
    letter-spacing: -.025em; line-height: 1.12; margin-bottom: .85rem;
}
.hero-text {
    color: rgba(226, 245, 243, .84); font-size: 1.02rem; line-height: 1.6;
    max-width: 50ch; margin: 0 auto;
}
.hero-art { display: block; margin: 1.2rem auto 0; width: min(330px, 80%); }

.section { text-align: center; margin: 2rem 0 1.1rem; }
.section .label { color: #5EEAD4; font-size: .72rem; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; }
.section .heading { color: #fff; font-size: 1.45rem; font-weight: 800; letter-spacing: -.02em; margin-top: .25rem; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: .9rem; }
.card {
    background: var(--glass); border: 1px solid var(--line); border-radius: 16px;
    padding: 1.25rem 1.1rem; text-align: center;
}
.card .icon {
    width: 46px; height: 46px; margin: 0 auto .7rem; border-radius: 50%;
    background: var(--accent-soft); border: 1px solid rgba(94, 234, 212, .35);
    display: flex; align-items: center; justify-content: center; font-size: 1.2rem;
}
.card .c-title { color: #fff; font-weight: 800; font-size: .98rem; margin-bottom: .3rem; }
.card .c-text { color: var(--muted); font-size: .86rem; line-height: 1.5; }

/* ---------- Boutons (questions d'exemple) ---------- */
.stButton > button {
    min-height: 3.4rem; padding: .6rem 1rem; border-radius: 14px;
    border: 1px solid var(--line); background: var(--glass); color: var(--ink);
    font-weight: 600; font-size: .9rem; text-align: left; justify-content: flex-start;
}
.stButton > button:hover { border-color: var(--accent); background: var(--accent-soft); color: #fff; }
.stButton > button:hover p { color: #fff; }
.stButton > button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

/* ---------- Messages ---------- */
[data-testid="stChatMessage"] {
    background: var(--glass); border: 1px solid var(--line); border-radius: 16px;
    padding: 1rem 1.2rem; margin-bottom: .8rem;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: linear-gradient(135deg, rgba(15, 118, 110, .85) 0%, rgba(10, 59, 73, .85) 100%);
    border-color: rgba(94, 234, 212, .35);
}
[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] li { font-size: .97rem; line-height: 1.65; }
[data-testid="stChatMessage"] pre { border-radius: 10px; }

/* ---------- Saisie ---------- */
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] textarea {
    background: #FFFFFF;
}
[data-testid="stChatInput"] {
    border-radius: 16px; border: 1px solid rgba(94, 234, 212, .5);
    box-shadow: 0 8px 24px rgba(0, 0, 0, .25);
}
[data-testid="stChatInput"]:focus-within { border-color: var(--accent); }
[data-testid="stChatInput"] textarea {
    color: #0F1F2B !important;
    -webkit-text-fill-color: #0F1F2B;
    caret-color: #0F766E;
    font-size: 1rem;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #6B7C8A !important;
    -webkit-text-fill-color: #6B7C8A;
    opacity: 1;
}

/* ---------- Sources ---------- */
[data-testid="stExpander"] {
    border: 1px solid var(--line); border-radius: 12px; background: rgba(255, 255, 255, .03); margin-top: .6rem;
}
[data-testid="stExpander"] summary { font-size: .85rem; font-weight: 700; }
.src { border: 1px solid var(--line); border-radius: 12px; padding: .7rem .85rem; margin: .5rem 0; background: var(--glass); }
.src-head { display: flex; align-items: center; gap: .5rem; margin-bottom: .35rem; }
.src-num {
    width: 20px; height: 20px; border-radius: 50%; background: var(--accent-soft); color: #99F6E4;
    font-size: .72rem; font-weight: 800; display: flex; align-items: center; justify-content: center;
}
.src-name { color: #fff; font-weight: 700; font-size: .82rem; }
.src-text { color: var(--muted); font-size: .8rem; line-height: 1.5; word-break: break-word; }

/* ---------- Barre latérale ---------- */
[data-testid="stSidebar"] { background: rgba(4, 26, 36, .94); border-right: 1px solid var(--line); }
.brand { display: flex; align-items: center; gap: .7rem; margin-bottom: .4rem; }
.brand img { width: 40px; height: 40px; }
.brand .name { color: #fff; font-weight: 800; font-size: 1.05rem; line-height: 1.1; }
.brand .sub { color: var(--muted); font-size: .78rem; }
[data-testid="stMetric"] { background: var(--glass); border: 1px solid var(--line); border-radius: 12px; padding: .6rem .8rem; }
[data-testid="stMetricLabel"] p { font-size: .74rem; color: var(--muted); }
[data-testid="stMetricValue"] { font-size: 1.3rem; font-weight: 700; color: #fff; }

@media (prefers-reduced-motion: reduce) {
    * { transition: none !important; animation: none !important; }
}
</style>
"""
st.markdown(html_block(STYLES), unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# 1. Chargement du pipeline RAG (une seule fois, mis en cache)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_rag():
    """Initialise la base vectorielle et le LLM (une seule fois, mis en cache).

    La configuration (config.yaml, clé Groq, chemin Chroma) est réutilisée depuis
    src/rag.py, qui reste inchangé. Les imports lourds sont faits ici pour que
    l'interface s'affiche avant le chargement de torch / des embeddings.
    """
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    load_dotenv()

    # rag.py charge config.yaml et .env à l'import (et lève une erreur si la clé manque)
    from rag import CHROMA_PATH, GROQ_API_KEY, config

    from langchain_chroma import Chroma
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings

    if not os.path.exists(CHROMA_PATH):
        return build_prompt, None, None

    embeddings = HuggingFaceEmbeddings(model_name=config["model"]["embedding_name"])
    vector_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    llm = ChatGroq(
        model=config["model"].get("llm_name", LLM_MODEL),
        temperature=config["model"]["temperature"],
        api_key=GROQ_API_KEY,
    )
    return build_prompt, vector_db, llm


# ----------------------------------------------------------------------------
# 2. Fonctions utilitaires
# ----------------------------------------------------------------------------
def stretch_button(label: str, **kwargs) -> bool:
    """Bouton pleine largeur, compatible avec toutes les versions de Streamlit."""
    try:
        return st.button(label, width="stretch", **kwargs)
    except TypeError:
        return st.button(label, use_container_width=True, **kwargs)


def plural(n: int, word: str) -> str:
    return f"{n} {word}{'s' if n > 1 else ''}"


def is_rate_limit(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return "ratelimit" in text or "rate limit" in text or "429" in text


def stream_answer(llm, prompt: str):
    for chunk in llm.stream(prompt):
        if chunk.content:
            yield chunk.content


def collect_sources(docs) -> tuple[list, list]:
    sources, excerpts, seen = [], [], set()
    for doc in docs:
        name = os.path.basename(doc.metadata.get("source", "Documentation Streamlit"))
        if name not in seen:
            seen.add(name)
            sources.append(name)
        text = doc.page_content.strip().replace("\n", " ")
        excerpts.append({"source": name, "text": text[:380] + ("…" if len(text) > 380 else "")})
    return sources, excerpts


def render_extras(msg: dict) -> None:
    """Extraits utilisés (expander) puis temps de réponse (caption)."""
    excerpts = msg.get("excerpts") or []
    if excerpts:
        n_extraits = len(excerpts)
        n_pages = len({ex["source"] for ex in excerpts})
        if msg.get("sources"):
            label = f"📚 Sources · {plural(n_extraits, 'extrait')} de {plural(n_pages, 'page')} de la documentation"
        else:
            label = f"🔎 Extraits consultés ({n_extraits})"
        with st.expander(label):
            cards = "".join(
                f'<div class="src"><div class="src-head"><span class="src-num">{i}</span>'
                f'<span class="src-name">{html.escape(ex["source"])}</span></div>'
                f'<div class="src-text">{html.escape(ex["text"])}</div></div>'
                for i, ex in enumerate(excerpts, start=1)
            )
            st.markdown(cards, unsafe_allow_html=True)
    if msg.get("latency") is not None:
        n = msg.get("n_docs", 0)
        st.caption(f"⏱ {msg['latency']:.1f} s · {plural(n, 'extrait')} consulté{'s' if n > 1 else ''}")


def render_message(msg: dict) -> None:
    avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_extras(msg)


def generate_reply(query: str, k: int, vector_db, llm, build_prompt) -> dict:
    """Retrieval + génération. À appeler dans un st.chat_message('assistant')."""
    start = time.perf_counter()

    with st.spinner(f"{BOT_NAME} consulte la documentation…"):
        docs = vector_db.similarity_search(query, k=k)

    if not docs:
        answer, sources, excerpts = NO_ANSWER, [], []
        st.markdown(answer)
    else:
        sources, excerpts = collect_sources(docs)
        context = "\n\n".join(d.page_content for d in docs)
        answer = st.write_stream(stream_answer(llm, build_prompt(context, query)))
        if NO_ANSWER_MARK in answer:
            sources = []  # pas de « sources » quand l'assistant refuse de répondre

    msg = {
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "excerpts": excerpts,
        "n_docs": len(docs),
        "latency": time.perf_counter() - start,
    }
    render_extras(msg)
    return msg


# Callbacks (exécutés avant le rechargement de la page : fiables)
def use_suggestion(text: str) -> None:
    st.session_state.pending_query = text


def clear_chat() -> None:
    st.session_state.messages = []
    st.session_state.pending_query = None


# ----------------------------------------------------------------------------
# 3. Blocs d'interface
# ----------------------------------------------------------------------------
def nav_html() -> str:
    return f"""
    <div class="nav">
        <div class="brand-inline"><img class="nav-logo" src="{LOGO_URI}" alt="">{APP_NAME}</div>
        <span class="status"><i></i>{BOT_NAME} est en ligne</span>
    </div>
    """


def render_hero() -> None:
    logo = f'<img src="{STREAMLIT_LOGO}" alt="">' if STREAMLIT_LOGO else ""
    st.markdown(
        html_block(
            f"""
            <div class="hero">
                {nav_html()}
                <div class="hero-body">
                    <span class="eyebrow">{logo}Assistant documentaire · Streamlit</span>
                    <div class="hero-title">Votre guide Streamlit,<br>prêt à répondre.</div>
                    <div class="hero-text">Posez une question en langage naturel : {BOT_NAME} cherche
                    la réponse dans la documentation officielle.</div>
                    <img class="hero-art" src="{HERO_ART}" alt="">
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def render_topbar() -> None:
    st.markdown(html_block(f'<div class="topbar">{nav_html()}</div>'), unsafe_allow_html=True)


def render_section(label: str, heading: str) -> None:
    st.markdown(
        html_block(
            f"""
            <div class="section">
                <div class="label">{label}</div>
                <div class="heading">{heading}</div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def render_features() -> None:
    render_section("Fonctionnalités", f"Ce que {BOT_NAME} fait pour vous")
    st.markdown(
        html_block(
            f"""
            <div class="cards">
                <div class="card">
                    <div class="icon">🔎</div>
                    <div class="c-title">Sources vérifiables</div>
                    <div class="c-text">Chaque réponse indique les extraits de la documentation utilisés.</div>
                </div>
                <div class="card">
                    <div class="icon">🛡️</div>
                    <div class="c-title">Fiable par conception</div>
                    <div class="c-text">Si l'information manque, {BOT_NAME} le dit au lieu d'inventer.</div>
                </div>
                <div class="card">
                    <div class="icon">⚡</div>
                    <div class="c-title">Rapide et transparent</div>
                    <div class="c-text">La réponse s'affiche en direct, avec son temps de génération.</div>
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# 4. État de session
# ----------------------------------------------------------------------------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("pending_query", None)

# ----------------------------------------------------------------------------
# 5. Barre latérale (partie haute)
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        html_block(
            f"""
            <div class="brand">
                <img src="{LOGO_URI}" alt="">
                <div>
                    <div class="name">{APP_NAME}</div>
                    <div class="sub">Assistant : {BOT_NAME}</div>
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("**Réglages**")
    top_k = st.slider(
        "Extraits consultés",
        min_value=1,
        max_value=8,
        value=3,
        help=(
            "Nombre d'extraits de la documentation donnés au modèle pour répondre. "
            "Plus il est élevé, plus la réponse a de contexte, mais plus elle peut être lente."
        ),
    )

# ----------------------------------------------------------------------------
# 6. Question (saisie ou question d'exemple cliquée)
# ----------------------------------------------------------------------------
typed_query = st.chat_input(f"Posez votre question à {BOT_NAME}…")
query = st.session_state.pending_query or typed_query
st.session_state.pending_query = None

landing = not st.session_state.messages and not query

# ----------------------------------------------------------------------------
# 7. En-tête + chargement
# ----------------------------------------------------------------------------
if landing:
    render_hero()
else:
    render_topbar()

try:
    with st.spinner("Chargement de la base de connaissances… (première ouverture uniquement)"):
        build_prompt, vector_db, llm = load_rag()
except ImportError as exc:
    st.error(f"Import impossible : {exc}")
    st.stop()
except Exception as exc:
    st.error(f"Initialisation impossible : {exc}")
    st.stop()

if vector_db is None:
    st.error("La base vectorielle est introuvable. Lancez d'abord `python3 src/ingest.py`.")
    st.stop()

# ----------------------------------------------------------------------------
# 8. Historique, ou questions d'exemple + fonctionnalités
# ----------------------------------------------------------------------------
for message in st.session_state.messages:
    render_message(message)

if landing:
    render_section("Exemples", "Par où commencer ?")
    cols = st.columns(2)
    for i, (icon, text) in enumerate(SUGGESTIONS):
        with cols[i % 2]:
            stretch_button(f"{icon}  {text}", key=f"suggestion_{i}", on_click=use_suggestion, args=(text,))
    render_features()

# ----------------------------------------------------------------------------
# 9. Génération
# ----------------------------------------------------------------------------
if query:
    user_msg = {"role": "user", "content": query}

    user_slot = st.empty()
    with user_slot.container():
        render_message(user_msg)
    assistant_slot = st.empty()
    notice = st.empty()

    try:
        with assistant_slot.container():
            with st.chat_message("assistant", avatar="🤖"):
                assistant_msg = generate_reply(query, top_k, vector_db, llm, build_prompt)
        st.session_state.messages += [user_msg, assistant_msg]
    except Exception as exc:
        # En cas d'échec, on retire la question sans réponse pour garder un historique propre
        user_slot.empty()
        assistant_slot.empty()
        if is_rate_limit(exc):
            notice.warning(
                "Le service de génération est momentanément saturé (limite de requêtes Groq). "
                "Patientez quelques secondes puis reposez votre question.",
                icon="⏳",
            )
            time.sleep(7)
            notice.empty()
        else:
            notice.error(f"La réponse n'a pas pu être générée : {exc}")

# ----------------------------------------------------------------------------
# 10. Barre latérale (partie basse : statistiques de session)
# ----------------------------------------------------------------------------
with st.sidebar:
    st.divider()
    st.markdown("**Session**")
    replies = [m for m in st.session_state.messages if m["role"] == "assistant"]
    latencies = [m["latency"] for m in replies if m.get("latency") is not None]
    c1, c2 = st.columns(2)
    c1.metric("Questions", len(replies))
    c2.metric("Latence moy.", f"{sum(latencies) / len(latencies):.1f} s" if latencies else "–")
    stretch_button("🗑️  Effacer la conversation", key="clear_chat", on_click=clear_chat)