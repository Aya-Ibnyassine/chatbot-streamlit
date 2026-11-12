# 🚀 Streamwise (AskStream.ai) - Pipeline RAG Industrialisé (MLOps)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?logo=langchain&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?logo=mlflow&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-945DD6?logo=dvc&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF?logo=github-actions&logoColor=white)

## 📌 Présentation du Projet
Ce projet académique est un système de question-réponse intelligent (RAG - Retrieval-Augmented Generation) conçu pour interagir avec la documentation officielle du framework Streamlit. 

Au-delà du simple chatbot, ce projet a été développé selon une approche **MLOps de bout en bout** pour assurer la reproductibilité, l'observabilité et la fiabilité du modèle en production.

## ✨ Fonctionnalités et Architecture MLOps
* 🧠 **Moteur RAG** : Modèle Llama3-8b (via l'API ultra-rapide Groq), orchestré par LangChain avec une base vectorielle ChromaDB et des embeddings HuggingFace.
* 🛡️ **Guardrails (Anti-Hallucination)** : Prompt engineering strict forçant l'IA à répondre *uniquement* via le contexte documentaire, ou à décliner poliment.
* 📊 **Suivi des Expérimentations (MLflow)** : Évaluation comparative des tailles de chunks (300 vs 500) en mesurant le F1-Score, le taux de récupération (Retrieval Rate) et la latence.
* 📦 **Versionnement des Données (DVC)** : Séparation stricte entre le code (Git) et les données textuelles/bases vectorielles (DVC).
* ⚙️ **Intégration Continue (CI/CD)** : Quality Gate automatisée via GitHub Actions pour valider l'intégrité de la configuration et des données de test à chaque `push`.
* 💻 **Interface UI Optimisée** : Application Streamlit modulaire, asynchrone (streaming) avec gestion de cache (`@st.cache_resource`) et télémétrie intégrée.

---

## 🛠️ Prérequis et Installation

### 1. Cloner le dépôt
```bash
git clone [https://github.com/votre-nom-dutilisateur/votre-repo.git](https://github.com/votre-nom-dutilisateur/votre-repo.git)
cd votre-repo
