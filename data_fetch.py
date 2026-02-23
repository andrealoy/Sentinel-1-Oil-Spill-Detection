import kaggle
import os 
import shutil  # Ajouté pour la suppression récursive
from pathlib import Path

def download_dataset(data_link, path, force_download=False):
    kaggle.api.authenticate()
    path = Path(path)

    if path.exists() and not force_download:
        # Vérifie si le dossier n'est pas vide avant de dire que c'est ok
        if any(path.iterdir()):
            print(f"✅ Dataset already exists at {path}. Skipping download.")
            return
    
    if path.exists() and force_download:
        print(f"Re-downloading: Removing existing files at {path}...")
        # shutil.rmtree est plus sûr pour supprimer des dossiers non vides
        shutil.rmtree(path)

    # Création du dossier (s'il a été supprimé ou n'existait pas)
    path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {data_link}...")
    kaggle.api.dataset_download_files(data_link, path=str(path), unzip=True)
    
    print("="*50)
    print(f"Done! Dataset location: {path.absolute()}")
    print("Folders found:", os.listdir(path))
    print("="*50)

if __name__ == "__main__":
    DATA_PATH = './data'
    DATASET_ID = "harikrishnacs/sentinel-1-sar-oil-spill-detection-dataset"
    download_dataset(DATASET_ID, DATA_PATH)



