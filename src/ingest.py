import os
import yaml
import mlflow
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Chargement du fichier de configuration YAML
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

RAW_DATA_PATH = config["data"]["raw_path"]
CHROMA_PATH = config["data"]["processed_path"]

def main():
    # Démarrage d'un run MLflow pour tracer l'ingestion
    mlflow.start_run(run_name="ingestion_pipeline")
    
    try:
        print("📥 1. Chargement des documents Markdown...")
        loader = DirectoryLoader(RAW_DATA_PATH, glob="**/*.md", loader_cls=TextLoader)
        documents = loader.load()
        print(f"✅ {len(documents)} documents chargés avec succès.")

        print("\n✂️ 2. Découpage du texte en chunks...")
        chunk_size = config["splitter"]["chunk_size"]
        chunk_overlap = config["splitter"]["chunk_overlap"]
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        print(f"✅ Texte découpé en {len(chunks)} chunks au total.")

        print("\n🧠 3. Initialisation des embeddings et création de la base vectorielle...")
        embedding_model_name = config["model"]["embedding_name"]
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PATH
        )
        
        # --- ENREGISTREMENT MLFLOW ---
        mlflow.log_param("chunk_size", chunk_size)
        mlflow.log_param("chunk_overlap", chunk_overlap)
        mlflow.log_param("embedding_model", embedding_model_name)
        mlflow.log_metric("num_documents", len(documents))
        mlflow.log_metric("num_chunks", len(chunks))
        # -----------------------------

        print(f"\n🎉 SUCCESS : Index vectoriel sauvegardé dans '{CHROMA_PATH}'.")
        print("📊 Paramètres et métriques enregistrés dans MLflow avec succès !")

    finally:
        mlflow.end_run()

if __name__ == "__main__":
    os.makedirs(CHROMA_PATH, exist_ok=True)
    main()