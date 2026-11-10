import os
import yaml
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

# Chargement du fichier de configuration YAML
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Chargement des variables d'environnement (depuis la racine si exécuté depuis src/)
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, "../.env")
load_dotenv(dotenv_path=env_path)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY introuvable dans le fichier .env")

CHROMA_PATH = config["data"]["processed_path"]

def main():
    print("🧠 Chargement de la base vectorielle ChromaDB...")

    embeddings = HuggingFaceEmbeddings(
        model_name=config["model"]["embedding_name"]
    )

    vector_db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    query = "How do I create a dataframe in Streamlit?"
    print(f"\n❓ Question posée : {query}\n")

    print("🔍 Recherche des chunks pertinents...")
    relevant_docs = vector_db.similarity_search(query, k=3)

    # Gestion des limites : si aucun document ne correspond
    if not relevant_docs:
        print("⚠️ Attention : Aucun document pertinent trouvé dans la base.")
        answer = "Je ne sais pas répondre à cette question avec la documentation fournie."
    else:
        context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

        print("🤖 Interrogation du LLM via Groq...")

        llm = ChatGroq(
            model="llama3-8b", # Ou ton modèle actif validé
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

    print("\n==============================================")
    print("🚀 RAG OPÉRATIONNEL !")
    print("==============================================\n")
    print(answer)

if __name__ == "__main__":
    main()