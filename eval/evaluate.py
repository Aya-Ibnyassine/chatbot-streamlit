import os
import glob
import json
import time
import yaml
import re
import mlflow
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

# 1. Chargement de la configuration
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

load_dotenv()
embeddings = HuggingFaceEmbeddings(model_name=config["model"]["embedding_name"])
vector_db = Chroma(persist_directory=config["data"]["processed_path"], embedding_function=embeddings)
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=config["model"]["temperature"])

# 2. Fonctions de nettoyage et de calcul (Tolérantes)
def clean_text(text):
    # Met en minuscules et supprime la ponctuation pour mieux comparer
    return re.sub(r'[^\w\s]', '', str(text).lower())

def calculate_f1(truth, prediction):
    truth_tokens = set(clean_text(truth).split())
    pred_tokens = set(clean_text(prediction).split())
    if not truth_tokens or not pred_tokens:
        return 0.0
    common = truth_tokens.intersection(pred_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    return 2 * (precision * recall) / (precision + recall)

# 3. Récupération des métriques de données (docs et chunks)
# Compte le nombre de fichiers Markdown dans ton dossier source
# Adapte le chemin "data/raw/" si tes documents sont ailleurs
raw_path = config["data"].get("raw_path", "data/raw/")
nbr_doc = len(glob.glob(os.path.join(raw_path, "*.md")))

# Compte le nombre exact de chunks dans ChromaDB
try:
    nbr_chunks = vector_db._collection.count()
except Exception:
    nbr_chunks = 0

# 4. Évaluation
questions = []
with open("eval/questions_test.jsonl", "r") as f:
    for line in f:
        questions.append(json.loads(line))

total_f1 = 0
total_latency = 0
retrieval_success = 0

print(f"Début de l'évaluation... (Chunks: {nbr_chunks}, Docs: {nbr_doc})")

for i, q in enumerate(questions):
    print(f"⏳ Traitement de la question {i+1}/{len(questions)}...")
    question = q["question"]
    ground_truth = q["ground_truth"]
    
    start_time = time.time()
    
    # Récupération
    docs = vector_db.similarity_search(question, k=3)
    context_text = "\n".join([doc.page_content for doc in docs])
    
    # Vérification de la récupération (on vérifie si des mots-clés de la réponse sont dans le contexte)
    truth_words = set(clean_text(ground_truth).split())
    context_words = set(clean_text(context_text).split())
    
    if truth_words.intersection(context_words) or "know" in ground_truth.lower():
        retrieval_success += 1
        
    # Génération
    if not docs:
        answer = "Je ne sais pas."
    else:
        prompt = f"Contexte:\n{context_text}\n\nQuestion: {question}\nRéponds uniquement avec les informations du contexte."
        response = llm.invoke(prompt)
        answer = response.content
        
    total_latency += (time.time() - start_time)
    total_f1 += calculate_f1(ground_truth, answer)

# Calcul des moyennes
avg_f1 = total_f1 / len(questions)
avg_latency = total_latency / len(questions)
retrieval_rate = retrieval_success / len(questions)

# 5. Envoi à MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("RAG_Streamlit_Evaluation")

with mlflow.start_run(run_name=f"ChunkSize_{config['splitter']['chunk_size']}"):
    # Paramètres
    mlflow.log_param("chunk_size", config['splitter']['chunk_size'])
    # Sauvegarder le fichier de configuration comme Artifact de cette expérience
    mlflow.log_artifact("configs/config.yaml")
    
    # Métriques
    mlflow.log_metric("nbr_doc", nbr_doc)
    mlflow.log_metric("nbr_chunks", nbr_chunks)
    mlflow.log_metric("avg_f1_score", avg_f1)
    mlflow.log_metric("avg_latency_seconds", avg_latency)
    mlflow.log_metric("retrieval_rate", retrieval_rate)

print(f"✅ Terminé ! F1: {avg_f1:.2f}, Latence: {avg_latency:.2f}s, Retrieval: {retrieval_rate:.2f}")