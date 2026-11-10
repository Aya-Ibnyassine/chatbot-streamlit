import os
import json
import yaml
import mlflow
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

# 1. Chargement de la configuration et de l'environnement
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, "../.env")
load_dotenv(dotenv_path=env_path)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CHROMA_PATH = config["data"]["processed_path"]

def evaluate_rag():
    print("🧪 Démarrage de l'évaluation du RAG...")
    
    # Initialisation des composants
    embeddings = HuggingFaceEmbeddings(model_name=config["model"]["embedding_name"])
    vector_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=config["model"]["temperature"], api_key=GROQ_API_KEY)

    # Chargement des questions de test
    test_file = "eval/questions_test.jsonl"
    if not os.path.exists(test_file):
        print(f"❌ Erreur : Le fichier {test_file} est introuvable.")
        return

    questions = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            questions.append(json.loads(line.strip()))

    # Lancement d'un run MLflow pour l'évaluation
    mlflow.start_run(run_name="rag_evaluation")
    
    correct_retrieval_count = 0
    total_questions = len(questions)

    for i, item in enumerate(questions):
        query = item["question"]
        print(f"\n--- Test {i+1}/{total_questions} : {query} ---")
        
        # Retrieval (Recherche des chunks)
        relevant_docs = vector_db.similarity_search(query, k=3)
        context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Gestion des limites : Vérification si le retrieval est vide ou non pertinent
        if not relevant_docs:
            answer = "Je ne sais pas car aucune information pertinente n'a été trouvée dans la documentation."
        else:
            prompt = f"""Réponds à la question en te basant uniquement sur le contexte fourni. Si la réponse n'y est pas, dis clairement que tu ne sais pas.

Contexte :
{context_text}

Question : {query}
Réponse :"""
            response = llm.invoke(prompt)
            answer = response.content

        print(f"🤖 Réponse du RAG :\n{answer}")
        
        # Validation simple (on considère le test réussi si le RAG a généré une réponse)
        if len(answer) > 10:
            correct_retrieval_count += 1

    # Calcul d'une métrique simple pour MLflow
    success_rate = (correct_retrieval_count / total_questions) * 100
    mlflow.log_metric("evaluation_success_rate", success_rate)
    mlflow.end_run()

    print(f"\n🎉 Évaluation terminée ! Taux de succès enregistré dans MLflow : {success_rate}%")

if __name__ == "__main__":
    evaluate_rag()