import html
import os

import streamlit as st
import yaml
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# ----------------------------------------------------------------------------
# 1. Configuration de la page (doit être le premier appel Streamlit)
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Assistant Streamlit | RAG",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded",
)

NO_ANSWER = "Je ne sais pas répondre à cette question avec la documentation fournie."
LLM_MODEL = "openai/gpt-oss-20b"

SUGGESTIONS = [
    "How do I create a dataframe in Streamlit?",
    "How can I cache data with st.cache_data?",
    "How do I use session state?",
    "How do I build a multipage app?",
]

# ----------------------------------------------------------------------------
# 2. Styles (CSS)
# ----------------------------------------------------------------------------
STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --ink: #1B2430;
    --muted: #667085;
    --line: #E3E7EC;
    --paper: #FFFFFF;
    --canvas: #F5F6F8;
    --accent: #0F766E;
    --accent-soft: #E3F2F0;
    --accent-ink: #0B5A54;
}

html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea {
    font-family: 'Manrope', -apple-system, 'Segoe UI', sans-serif !important;
}

.stApp { background: var(--canvas); color: var(--ink); }

/* Interface Streamlit : on retire le bruit, on garde le bouton de la sidebar */
#MainMenu, footer { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { visibility: hidden; }

.block-container {
    max-width: 820px;
    padding-top: 2.2rem;
    padding-bottom: 6rem;
}

/* ---------- En-tête ---------- */
.app-header { margin-bottom: 1.6rem; }
.app-header h1 {
    font-size: 1.9rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: var(--ink);
    margin: 0 0 .35rem 0;
    padding: 0;
}
.app-header p {
    color: var(--muted);
    font-size: 1rem;
    line-height: 1.55;
    margin: 0 0 .9rem 0;
    max-width: 60ch;
}
.pill {
    display: inline-flex;
    align-items: center;
    gap: .45rem;
    padding: .28rem .7rem;
    border-radius: 999px;
    background: var(--accent-soft);
    color: var(--accent-ink);
    font-size: .8rem;
    font-weight: 600;
}
.pill .dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--accent);
}

/* ---------- Messages ---------- */
[data-testid="stChatMessage"] {
    background: var(--paper);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 1rem 1.15rem;
    margin-bottom: .85rem;
    box-shadow: 0 1px 2px rgba(16, 24, 40, .04);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: var(--accent-soft);
    border-color: #CBE4E0;
}
[data-testid="stChatMessage"] p { line-height: 1.65; }
[data-testid="stChatMessage"] pre { border-radius: 10px; }

/* ---------- Sources ---------- */
.sources { margin-top: .9rem; display: flex; flex-wrap: wrap; align-items: center; gap: .4rem; }
.sources .label { color: var(--muted); font-size: .8rem; font-weight: 600; margin-right: .2rem; }
.chip {
    padding: .2rem .6rem;
    border: 1px solid var(--line);
    border-radius: 8px;
    background: var(--canvas);
    color: var(--ink);
    font-size: .78rem;
    font-weight: 500;
}

/* ---------- Boutons (suggestions, effacer) ---------- */
.stButton > button {
    width: 100%;
    text-align: left;
    justify-content: flex-start;
    padding: .7rem .95rem;
    border-radius: 10px;
    border: 1px solid var(--line);
    background: var(--paper);
    color: var(--ink);
    font-weight: 500;
    font-size: .9rem;
    transition: border-color .15s ease, background .15s ease;
}
.stButton > button:hover {
    border-color: var(--accent);
    background: var(--accent-soft);
    color: var(--accent-ink);
}
.stButton > button:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
}

.section-title {
    font-size: .95rem;
    font-weight: 700;
    color: var(--ink);
    margin: 1.4rem 0 .6rem 0;
}

/* ---------- Champ de saisie ---------- */
[data-testid="stChatInput"] {
    border-radius: 14px;
    border: 1px solid var(--line);
    background: var(--paper);
    box-shadow: 0 4px 16px rgba(16, 24, 40, .06);
}
[data-testid="stChatInput"]:focus-within { border-color: var(--accent); }

/* ---------- Expander ---------- */
[data-testid="stExpander"] {
    border: 1px solid var(--line);
    border-radius: 10px;
    background: var(--paper);
    margin-top: .6rem;
}
[data-testid="stExpander"] summary { font-size: .85rem; font-weight: 600; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: var(--paper);
    border-right: 1px solid var(--line);
}
.brand { display: flex; align-items: center; gap: .7rem; margin-bottom: .4rem; }
.brand .logo {
    width: 38px; height: 38px; border-radius: 10px;
    background: var(--accent);
    color: #fff;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.15rem;
}
.brand .name { font-weight: 800; font-size: 1.02rem; line-height: 1.1; }
.brand .sub { color: var(--muted); font-size: .78rem; }

.side-title { font-weight: 700; font-size: .9rem; margin: 1.3rem 0 .5rem 0; }
.kv {
    display: flex; justify-content: space-between; gap: 1rem;
    padding: .42rem 0;
    border-bottom: 1px solid var(--line);
    font-size: .84rem;
}
.kv:last-child { border-bottom: none; }
.kv .k { color: var(--muted); }
.kv .v { font-weight: 600; text-align: right; word-break: break-word; }

.guard {
    display: flex; gap: .6rem;
    padding: .7rem .8rem;
    border-radius: 10px;
    background: var(--accent-soft);
    color: var(--accent-ink);
    font-size: .82rem;
    line-height: 1.45;
}

@media (prefers-reduced-motion: reduce) {
    * { transition: none !important; animation: none !important; }
}
</style>
"""
st.markdown(STYLES, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# 3. Configuration, base vectorielle et LLM (mis en cache)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_config_and_env():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(dotenv_path=os.path.join(current_dir, "../.env"))

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        load_dotenv()
        groq_api_key = os.getenv("GROQ_API_KEY")

    return config, groq_api_key


config, GROQ_API_KEY = load_config_and_env()
CHROMA_PATH = config["data"]["processed_path"]


@st.cache_resource
def init_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name=config["model"]["embedding_name"])
    if not os.path.exists(CHROMA_PATH):
        return None
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)


@st.cache_resource
def init_llm(api_key: str, temperature: float):
    return ChatGroq(model=LLM_MODEL, temperature=temperature, api_key=api_key)


vector_db = init_vector_db()


# ----------------------------------------------------------------------------
# 4. Fonctions utilitaires
# ----------------------------------------------------------------------------
def kv_row(key: str, value: str) -> str:
    return (
        f'<div class="kv"><span class="k">{html.escape(key)}</span>'
        f'<span class="v">{html.escape(str(value))}</span></div>'
    )


def render_sources(sources: list[str], excerpts: list[dict]) -> None:
    """Affiche les sources sous forme de pastilles + les extraits utilisés."""
    if not sources:
        return
    chips = "".join(f'<span class="chip">{html.escape(s)}</span>' for s in sources)
    st.markdown(
        f'<div class="sources"><span class="label">Sources</span>{chips}</div>',
        unsafe_allow_html=True,
    )
    if excerpts:
        with st.expander("Voir les extraits utilisés"):
            for i, ex in enumerate(excerpts, start=1):
                st.markdown(f"**{i}. {ex['source']}**")
                st.caption(ex["text"])


def render_message(message: dict) -> None:
    avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []), message.get("excerpts", []))


def stream_answer(llm, prompt: str):
    for chunk in llm.stream(prompt):
        if chunk.content:
            yield chunk.content


def build_prompt(context_text: str, query: str) -> str:
    return f"""Tu es un assistant technique expert sur Streamlit. Réponds à la question en te basant **uniquement** sur le contexte fourni ci-dessous.
Si l'information n'est pas présente dans le contexte ou si la question est hors du sujet de Streamlit, réponds strictement : "{NO_ANSWER}" N'invente rien.

Contexte :
{context_text}

Question : {query}
Réponse :"""


# ----------------------------------------------------------------------------
# 5. Barre latérale
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="logo">💬</div>
            <div>
                <div class="name">Assistant Streamlit</div>
                <div class="sub">Recherche dans la documentation</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="side-title">Configuration</div>', unsafe_allow_html=True)
    st.markdown(
        kv_row("Modèle", LLM_MODEL)
        + kv_row("Embeddings", config["model"]["embedding_name"].split("/")[-1])
        + kv_row("Taille des chunks", config["splitter"]["chunk_size"])
        + kv_row("Chevauchement", config["splitter"]["chunk_overlap"]),
        unsafe_allow_html=True,
    )

    st.markdown('<div class="side-title">Recherche</div>', unsafe_allow_html=True)
    top_k = st.slider(
        "Nombre de passages consultés",
        min_value=1,
        max_value=8,
        value=3,
        help="Plus il est élevé, plus la réponse s'appuie sur de contexte, mais plus elle peut être lente.",
    )

    st.markdown('<div class="side-title">Protections</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="guard"><span>🛡️</span><span>Les réponses se basent uniquement sur la documentation. '
        "Hors sujet ou information absente : l'assistant le dit au lieu d'inventer.</span></div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    if st.button("Effacer la conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ----------------------------------------------------------------------------
# 6. En-tête
# ----------------------------------------------------------------------------
db_ready = vector_db is not None and bool(GROQ_API_KEY)
status_text = "Base de connaissances prête" if db_ready else "Configuration incomplète"

st.markdown(
    f"""
    <div class="app-header">
        <h1>Posez vos questions sur Streamlit</h1>
        <p>L'assistant cherche dans la documentation officielle et répond à partir
        des passages trouvés, avec leurs sources.</p>
        <span class="pill"><span class="dot"></span>{status_text}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if vector_db is None:
    st.error(
        f"La base vectorielle est introuvable dans `{CHROMA_PATH}`. "
        "Lancez d'abord `python3 src/ingest.py`."
    )
    st.stop()

if not GROQ_API_KEY:
    st.error("La clé `GROQ_API_KEY` est introuvable. Ajoutez-la dans le fichier `.env`.")
    st.stop()


# ----------------------------------------------------------------------------
# 7. Historique et suggestions
# ----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

for message in st.session_state.messages:
    render_message(message)

if not st.session_state.messages:
    st.markdown('<div class="section-title">Pour commencer</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, suggestion in enumerate(SUGGESTIONS):
        if cols[i % 2].button(suggestion, key=f"suggestion_{i}"):
            st.session_state.pending_query = suggestion
            st.rerun()

# ----------------------------------------------------------------------------
# 8. Saisie et génération de la réponse
# ----------------------------------------------------------------------------
query = st.chat_input("Posez votre question, par exemple : How do I create a dataframe?")

if not query and st.session_state.pending_query:
    query = st.session_state.pending_query
    st.session_state.pending_query = None

if query:
    user_msg = {"role": "user", "content": query}
    st.session_state.messages.append(user_msg)
    render_message(user_msg)

    with st.chat_message("assistant", avatar="🤖"):
        sources: list[str] = []
        excerpts: list[dict] = []

        with st.spinner("Recherche dans la documentation…"):
            relevant_docs = vector_db.similarity_search(query, k=top_k)

        if not relevant_docs:
            answer = NO_ANSWER
            st.markdown(answer)
        else:
            context_text = "\n\n".join(doc.page_content for doc in relevant_docs)

            seen = set()
            for doc in relevant_docs:
                name = os.path.basename(doc.metadata.get("source", "Documentation Streamlit"))
                if name not in seen:
                    seen.add(name)
                    sources.append(name)
                text = doc.page_content.strip().replace("\n", " ")
                excerpts.append(
                    {"source": name, "text": text[:350] + ("…" if len(text) > 350 else "")}
                )

            llm = init_llm(GROQ_API_KEY, config["model"]["temperature"])
            try:
                answer = st.write_stream(stream_answer(llm, build_prompt(context_text, query)))
            except Exception as exc:  # erreur réseau, quota Groq, etc.
                answer = "La réponse n'a pas pu être générée. Réessayez dans un instant."
                st.error(f"{answer} (détail : {exc})")
                sources, excerpts = [], []

            # Pas de sources affichées quand l'assistant refuse de répondre
            if "Je ne sais pas" in answer:
                sources, excerpts = [], []

            render_sources(sources, excerpts)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources, "excerpts": excerpts}
    )