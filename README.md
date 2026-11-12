# Streamwise - Pipeline RAG Industrialisé (MLOps)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?logo=langchain&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?logo=mlflow&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-945DD6?logo=dvc&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF?logo=github-actions&logoColor=white)

## Présentation du Projet
Ce projet académique est un système de question-réponse intelligent (RAG - Retrieval-Augmented Generation) conçu pour interagir avec la documentation officielle du framework Streamlit. 

Au-delà du simple agent conversationnel, ce projet a été développé selon une approche **MLOps de bout en bout** afin d'assurer la reproductibilité, l'observabilité et la fiabilité du modèle en production.

## Architecture et Composants MLOps
* **Moteur RAG** : Modèle Llama3-8b (via l'API Groq), orchestré par LangChain avec une base vectorielle ChromaDB et des embeddings HuggingFace.
* **Gouvernance et Guardrails (Anti-Hallucination)** : Prompt engineering strict forçant le LLM à répondre uniquement via le contexte documentaire, ou à décliner formellement.
* **Suivi des Expérimentations (MLflow)** : Évaluation comparative des paramètres d'ingestion (chunk size de 300 vs 500) en mesurant le F1-Score, le taux de récupération (Retrieval Rate) et la latence moyenne.
* **Versionnement des Données (DVC)** : Séparation architecturale stricte entre le code source (Git) et les données textuelles/bases vectorielles (DVC).
* **Intégration Continue (CI/CD)** : Quality Gate automatisée via GitHub Actions pour valider l'intégrité de la configuration (YAML) et des jeux de données d'évaluation à chaque push.
* **Interface UI Optimisée** : Application Streamlit modulaire, asynchrone (streaming) avec gestion de cache (`@st.cache_resource`), calcul de latence en temps réel et gestion des quotas d'API.

---

## Prérequis et Installation

### 1. Cloner le dépôt
```bash
git clone [https://github.com/votre-nom-dutilisateur/votre-repo.git](https://github.com/votre-nom-dutilisateur/votre-repo.git)
cd votre-repo
```

### 2. Environnement virtuel et dépendances
```bash
python3 -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configuration des variables d'environnement
Créez un fichier `.env` à la racine du projet et ajoutez votre clé d'authentification API :
```env
GROQ_API_KEY=votre_cle_api_ici
```

---

## Utilisation du Pipeline

### Étape 1 : Récupération des données (DVC)
*(Nécessaire uniquement si les données sont stockées sur un serveur distant via DVC)*
```bash
dvc pull
```

### Étape 2 : Ingestion et Vectorisation
Ce script découpe les documents Markdown sources et génère la base de données vectorielle ChromaDB.
```bash
python src/ingest.py
```

### Étape 3 : Évaluation du Modèle (MLflow)
Lance l'évaluation sur le jeu de données de test (Golden Dataset `questions_test.jsonl`).
```bash
python eval/evaluate.py
```
*Pour visualiser les résultats des expérimentations, lancez `mlflow ui` et ouvrez `http://127.0.0.1:5000`.*

### Étape 4 : Lancement de l'Application Web
Démarre l'interface utilisateur Streamwise.
```bash
streamlit run src/app.py
```

---

## Structure du Dépôt

```text
├── .github/workflows/   # Pipeline CI/CD (GitHub Actions)
├── configs/             # Fichiers de configuration globaux (config.yaml)
├── data/
│   ├── raw/             # Documentation Streamlit brute (.md)
│   └── processed/       # Base vectorielle ChromaDB (ignorée par Git, gérée par DVC)
├── eval/                # Scripts d'évaluation et dataset de test (.jsonl)
├── src/                 # Code source Python
│   ├── app.py           # Interface utilisateur Streamlit
│   ├── ingest.py        # Pipeline d'ingestion et text splitting
│   └── rag.py           # Logique métier du RAG et de génération
├── tests/               # Tests unitaires pour la Quality Gate (test_pipeline.py)
├── data.dvc             # Pointeurs de versionnement des données
└── requirements.txt     # Dépendances Python
```

---

## Auteur
**Aya Ibnyassine**  
*Élève Ingénieure en Data Science et Intelligence Artificielle - ENSA Fès*
