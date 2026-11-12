import json
import os

def test_golden_dataset_format_and_volume():
    """Vérifie que le dataset d'évaluation est valide et contient assez de questions."""
    dataset_path = "eval/questions_test.jsonl"
    
    # Le test échoue immédiatement si le fichier a été supprimé par erreur
    assert os.path.exists(dataset_path), f"Le fichier {dataset_path} est introuvable !"
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    # Vérification du volume minimum (exigé pour une bonne évaluation)
    assert len(lines) >= 10, "Le dataset doit contenir au moins 10 questions pour être statistiquement pertinent."
    
    # Vérification du format strict attendu par le script d'évaluation
    for i, line in enumerate(lines):
        data = json.loads(line)
        assert "question" in data, f"Ligne {i+1} : Clé 'question' manquante."
        assert "ground_truth" in data, f"Ligne {i+1} : Clé 'ground_truth' manquante."