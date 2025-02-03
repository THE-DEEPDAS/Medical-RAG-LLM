import os
import wget
from typing import Optional

def download_model(url: str, output_path: str, force: bool = False) -> Optional[str]:
    """Download a model file if it doesn't exist locally"""
    try:
        if os.path.exists(output_path) and not force:
            print(f"Model already exists at {output_path}")
            return output_path
            
        print(f"Downloading model from {url}...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        filename = wget.download(url, output_path)
        print(f"\nModel downloaded successfully to {filename}")
        return filename
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        return None

def ensure_model_exists():
    """Ensure the required model file exists"""
    model_path = "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
    model_url = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
    
    return download_model(model_url, model_path)
