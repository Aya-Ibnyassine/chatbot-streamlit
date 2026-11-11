import os
import yaml

def test_config_exists():
    """Vérifie que le fichier de configuration MLOps est bien présent."""
    assert os.path.exists("configs/config.yaml"), "Le fichier config.yaml est introuvable !"

def test_config_has_chunk_size():
    """Vérifie que la configuration contient bien le paramètre chunk_size."""
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    assert "chunk_size" in config["splitter"], "Le paramètre chunk_size est manquant !"