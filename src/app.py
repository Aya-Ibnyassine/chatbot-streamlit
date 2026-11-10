import os
import yaml
from dotenv import load_dotenv
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

# 1. Configuration de la page Streamlit (Design & Titre)
st.set_page_config(
    page_title="Streamlit RAG Assistant | MLOps",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. Chargement de la configuration YAML et des variables d'environnement
@st.cache_resource
def load_config_and_env():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(current_dir, "../.env")
    load_dotenv(dotenv_path=env_path)
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        # Essayer aussi à la racine directe si lancé depuis la racine
        load_dotenv()
        groq_api_key = os.getenv("GROQ_API_KEY")
        
    return config, groq_api_key

config, GROQ_API_KEY = load_config_and_env()

CHROMA_PATH = config["data"]["processed_path"]

# 3. Chargement du modèle d'embeddings et de la base vectorielle (avec cache pour la performance)
@st.cache_resource
def init_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name=config["model"]["embedding_name"])
    if not os.path.exists(CHROMA_PATH):
        return None
    vector_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    return vector_db

vector_db = init_vector_db()

# --- SIDEBAR (Barre latérale esthétique) ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=70)
    st.markdown("### ⚙️ Paramètres MLOps")
    st.markdown("---")
    
    st.markdown(f"**Modèle LLM :** `openai/gpt-oss-20b`")
    st.markdown(f"**Embeddings :** `{config['model']['embedding_name']}`")
    st.markdown(f"**Chunk Size :** `{config['splitter']['chunk_size']}`")
    st.markdown(f"**Chunk Overlap :** `{config['splitter']['chunk_overlap']}`")
    
    st.markdown("---")
    st.markdown("### 🛡️ Guardrails")
    st.success("Actifs (Anti-hallucination & Refus hors périmètre)")
    
    st.markdown("---")
    if st.button("🗑️ Effacer l'historique"):
        st.session_state.messages = []
        st.rerun()

# --- PAGE PRINCIPALE ---
st.title("💬 Assistant Intelligent - Documentation Streamlit")
st.markdown("Posez vos questions sur Streamlit. L'assistant interroge la base vectorielle ChromaDB via un pipeline RAG sécurisé.")

# Vérification de la présence de la base vectorielle
if vector_db is None:
    st.error(f"⚠️ La base vectorielle est introuvable dans `{CHROMA_PATH}`. Veuillez d'abord exécuter `python3 src/ingest.py` !")
    st.stop()

if not GROQ_API_KEY:
    st.error("⚠️ La clé `GROQ_API_KEY` est introuvable dans le fichier `.env`.")
    st.stop()

# Initialisation de l'historique de chat dans la session Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Bonjour ! Je suis votre assistant MLOps RAG. Comment puis-je vous aider avec Streamlit aujourd'hui ?"}
    ]

# Affichage de l'historique des messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Champ de saisie utilisateur
if query := st.chat_input("Ex: How do I create a dataframe in Streamlit?"):
    # Ajouter le message de l'utilisateur à l'historique
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Génération de la réponse
    with st.chat_message("assistant"):
        with st.spinner("🔍 Recherche des documents et génération de la réponse..."):
            
            # 1. Retrieval (Recherche des k chunks les plus proches)
            relevant_docs = vector_db.similarity_search(query, k=3)
            
            # 2. Gestion des limites / Guardrails
            if not relevant_docs:
                answer = "Je ne sais pas répondre à cette question avec la documentation fournie (aucun document pertinent trouvé)."
                sources_str = ""
            else:
                context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
                
                # Récupération des sources (nom des fichiers Markdown d'origine)
                sources = set()
                for doc in relevant_docs:
                    source_path = doc.metadata.get("source", "Documentation Streamlit")
                    sources.add(os.path.basename(source_path))
                sources_str = ", ".join(sources)

                # 3. Appel au LLM via Groq
                llm = ChatGroq(
                    model="openai/gpt-oss-20b",
                    temperature=config["model"]["temperature"],
                    api_key=GROQ_API_KEY
                )

                prompt = f"""Tu es un assistant technique expert sur Streamlit. Réponds à la question en te basant **uniquement** sur le contexte fourni ci-dessous. 
Si l'information n'est pas présente dans le contexte ou si la question est hors du sujet de Streamlit, réponds strictement : "Je ne sais pas répondre à cette question avec la documentation fournie." N'invente rien.

Contexte :
{context_text}

Question : {query}
Réponse :"""

                response = llm.invoke(prompt)
                answer = response.content
                
                # Ajout discret des sources à la fin de la réponse
                if "Je ne sais pas" not in answer:
                    answer += f"\n\n*📚 Source(s) : {sources_str}*"

            st.markdown(answer)
            
    # Sauvegarde de la réponse dans l'historique
    st.session_state.messages.append({"role": "assistant", "content": answer})