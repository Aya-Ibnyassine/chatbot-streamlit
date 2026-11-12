import yaml
import os

def test_chunk_overlap_must_be_smaller_than_size():
    """Vérifie que le chevauchement est strictement inférieur à la taille du chunk."""
    config_path = "configs/config.yaml"
    assert os.path.exists(config_path), "Le fichier config.yaml est introuvable."
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    chunk_size = config["splitter"]["chunk_size"]
    chunk_overlap = config["splitter"]["chunk_overlap"]
    
    assert chunk_overlap < chunk_size, f"L'overlap ({chunk_overlap}) doit être inférieur au chunk_size ({chunk_size}) !"

def test_raw_data_folder_exists():
    """Vérifie que le dossier des données brutes existe bien avant l'ingestion."""
    assert os.path.exists("data/raw"), "Le dossier data/raw n'existe pas."