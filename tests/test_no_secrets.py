import os
import yaml

def test_no_api_key_hardcoded_in_config():
    """Vérifie qu'aucune clé API Groq n'est codée en dur dans la configuration."""
    config_path = "configs/config.yaml"
    
    if not os.path.exists(config_path):
        return  # Si le fichier n'existe pas, un autre test (test_pipeline) s'en chargera
        
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read().lower()
        
    # Les clés Groq commencent généralement par "gsk_"
    assert "gsk_" not in content, "ALERTE SÉCURITÉ : Une clé API semble être codée en dur dans config.yaml !"
    
def test_env_file_is_ignored():
    """Vérifie que le fichier .env est bien déclaré dans le .gitignore."""
    gitignore_path = ".gitignore"
    
    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert ".env" in content, "ALERTE SÉCURITÉ : Le fichier .env n'est pas ignoré dans le .gitignore !"