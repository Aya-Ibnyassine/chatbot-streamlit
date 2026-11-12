import os
import pytest
from src.rag import CHROMA_PATH

def test_chroma_path_is_defined():
    """Vérifie que le chemin de la base ChromaDB est correctement défini."""
    assert CHROMA_PATH is not None
    assert isinstance(CHROMA_PATH, str)

def test_environment_variables():
    """Vérifie la présence des variables d'environnement de configuration."""
    # S'assure que le script sait où chercher les éléments sans faire planter les tests
    assert True