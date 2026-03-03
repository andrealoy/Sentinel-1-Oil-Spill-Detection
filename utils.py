########################utils.py ######################################################################
#                                                                                                     #
# A collection of utility functions for image dataset management, visualization, and model evaluation.#
# Author: Rose Aupepin , Andréa Loy , Alizée Robin, Omar Zeroual                                      #  
#                                                                                                     #
#######################################################################################################

# Standard library imports
import os
import random
import shutil
import hashlib
import imghdr
import torch


# Data manipulation and numerical arrays
import numpy as np
import pandas as pd
import seaborn as sns

# Visualization
import matplotlib.pyplot as plt
import plotly.express as px

import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pathlib import Path
from PIL import Image

# Machine Learning and Deep Learning
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    precision_score, 
    recall_score, 
    f1_score
)

from typing import Union

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from collections import Counter
import os
import shutil
import random
from pathlib import Path
from collections import Counter

def create_spatial_split_full(data_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    random.seed(seed)
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    # 1. Analyse des régions (Correction ici : on utilise rglob comme dans ton count)
    region_map = {} 
    all_images_count = 0
    
    print(f"🔍 Analyse du dossier source : {data_dir}")
    
    # On cherche tous les fichiers .jpg (ou * si tu as plusieurs formats)
    # On remplace la boucle class_folder par rglob
    for img_path in data_dir.rglob('*.jpg'): 
        if img_path.is_file() and not img_path.name.startswith('.'):
            region = extract_scene_id(img_path.stem)
            if region not in region_map:
                region_map[region] = []
            region_map[region].append(img_path)
            all_images_count += 1

    if all_images_count == 0:
        print("❌ AUCUNE IMAGE TROUVÉE ! Vérifie l'extension (.jpg ou .png ?) ou le chemin.")
        return

    # 2. Tri et Répartition (Le reste de ta logique est bon)
    sorted_regions = sorted(region_map.keys(), key=lambda r: len(region_map[r]), reverse=True)
    
    train_regions, val_regions, test_regions = [], [], []
    train_count, val_count, test_count = 0, 0, 0
    
    target_val = all_images_count * val_ratio
    target_test = all_images_count * test_ratio

    for r in sorted_regions:
        count = len(region_map[r])
        if count > target_val and count > target_test:
            train_regions.append(r)
            train_count += count
        elif val_count + count <= target_val * 1.5:
            val_regions.append(r)
            val_count += count
        elif test_count + count <= target_test * 1.5:
            test_regions.append(r)
            test_count += count
        else:
            train_regions.append(r)
            train_count += count

    print(f"\n📊 Répartition finale : {train_count} Train | {val_count} Val | {test_count} Test")

    # 3. Copie (Correction : img_path.parent.name récupère le nom du dossier de classe)
    print("🚀 Copie en cours...")
    for split_name, regions in [('train', train_regions), ('val', val_regions), ('test', test_regions)]:
        for r in regions:
            for img_path in region_map[r]:
                class_name = img_path.parent.name 
                dest_dir = output_dir / split_name / class_name
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_path, dest_dir / img_path.name)

    print(f"✅ Terminé ! {all_images_count} images traitées.")

def count_images_by_region(data_dir):
    """
    Compte le nombre d'images par région (ex: VEN, GBR, BAH) en extrayant le code géographique du nom de fichier.
    Affiche les résultats sous forme de graphique à barres.
    """
    data_dir = Path(data_dir)
    region_counts = Counter()

    for img_path in data_dir.rglob("*.jpg"):
        scene_id = extract_scene_id(img_path.stem)  # Extrait le code géographique
        region_counts[scene_id] += 1

    # Affichage des résultats
    regions = list(region_counts.keys())
    counts = list(region_counts.values())

    print("Nombre d'images par région:")
    for region, count in region_counts.items():
        print(f"{region}: {count} images.")

def find_and_remove_black_images(data_dir, threshold=1.0, delete=False):
    """
    Parcourt le dossier et trouve les images presque totalement noires.
    - threshold: si l'écart-type des pixels est inférieur à ça, on considère l'image noire.
    """
    data_dir = Path(data_dir)
    black_images = []
    
    # On cherche tous les jpg/png dans tous les sous-dossiers
    for img_path in data_dir.rglob("*.jpg"):
        # On lit l'image en niveaux de gris
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            # Si l'image n'a aucune variation (tous les pixels sont les mêmes)
            if np.std(img) < threshold:
                black_images.append(img_path)
                
    print(f"🧹 Nombre d'images noires détectées : {len(black_images)}")
    
    if delete:
        for img_path in black_images:
            os.remove(img_path)
        print("🗑️ Toutes les images noires ont été supprimées !")
        
    return black_images

def visualize_random_gradcam(model, target_layer, dataset, device, num_images=5):
    """
    Affiche les cartes d'activation Grad-CAM pour des images tirées au hasard.
    
    Paramètres:
    - model : Ton modèle entraîné (ResNet ou SimpleCNN)
    - target_layer : La dernière couche de convolution (ex: model.layer4[-1])
    - dataset : Ton jeu de données de test (ex: test_dataset)
    - device : 'cuda' ou 'cpu'
    - num_images : Le nombre de paires d'images à afficher
    """
    model.eval()
    
    # 1. Initialiser le Grad-CAM (il attend une liste de couches, on gère ça ici)
    cam = GradCAM(model=model, target_layers=[target_layer])
    
    # 2. Tirer des indices au hasard dans le dataset
    indices = random.sample(range(len(dataset)), num_images)
    
    # Préparer la grille d'affichage
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 4 * num_images))
    if num_images == 1:
        axes = [axes] # Sécurité si on ne demande qu'une seule image
        
    for i, idx in enumerate(indices):
        # Récupérer l'image et son vrai label
        img_tensor, true_label = dataset[idx]
        
        # Préparer le tenseur pour le modèle (ajouter la dimension du batch)
        input_tensor = img_tensor.unsqueeze(0).to(device)
        
        # 3. Générer le masque Grad-CAM (sur la prédiction par défaut)
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
        grayscale_cam = grayscale_cam[0, :]
        
        # 4. Obtenir la prédiction pour le code couleur
        with torch.no_grad():
            output = model(input_tensor)
            pred_class = torch.argmax(output, dim=1).item()
            
        color = "green" if pred_class == true_label else "red"
        
        # 5. Préparer l'image pour l'affichage (dé-normaliser)
        orig_img = img_tensor.squeeze().cpu().numpy() # [128, 128]
        img_rgb = np.stack((orig_img,)*3, axis=-1)    # [128, 128, 3]
        img_rgb = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min() + 1e-8) # 0 à 1
        
        # Superposer
        visualization = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)
        
        # 6. Affichage
        axes[i][0].imshow(orig_img, cmap='gray')
        axes[i][0].set_title(f"Originale (Vraie Classe: {true_label})")
        axes[i][0].axis('off')
        
        axes[i][1].imshow(visualization)
        axes[i][1].set_title(f"Grad-CAM (Prédit: {pred_class})", color=color, fontweight='bold')
        axes[i][1].axis('off')
        
    plt.tight_layout()
    plt.show()

def extract_scene_id(filename_stem):
    """
    Extrait le code géographique (ex: VEN, GBR, BAH) du nom du fichier.
    Attend un nom sans extension comme '0_0_0_img_1TDZ4mPosbER_VEN_cls_0'
    """
    parts = filename_stem.split('_')
    
    # Sécurité : on s'assure que le fichier correspond bien au format attendu
    if len(parts) >= 3:
        # Le code région (VEN, GBR...) est toujours l'avant-avant-dernier bloc
        return parts[-3] 
    else:
        return "UNKNOWN"

def set_seed(seed=42):
    """Fixe toutes les graines aléatoires pour rendre l'entraînement reproductible."""
    # 1. Python de base et OS
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 2. Numpy
    np.random.seed(seed)
    
    # 3. PyTorch (CPU et GPU)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # Pour le multi-GPU
        
    # 4. Forcer le comportement déterministe de la carte graphique (CUDNN)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Graine aléatoire fixée à {seed} 🌱")

def plot_pytorch_history(history):
    epochs_range = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(15, 5))

    # --- Graphique de la Perte (Loss) ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Training', color='#1f77b4', linewidth=2)
    plt.plot(epochs_range, history['val_loss'], label='Validation', color='#ff7f0e', linestyle='--')
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- Graphique du F1-Score ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['val_f1'], label='Validation F1', color='#2ca02c', linewidth=2)
    plt.title('Validation F1 Score', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def get_mean_and_std(loader):
    mean = 0.
    std = 0.
    total_images_count = 0
    
    for images, _ in loader:
        # Batch size (B, C, H, W)
        batch_samples = images.size(0) 
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count
    
    return mean, std

def evaluate_test_set(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Affichage du rapport complet (Precision, Recall, F1-score)
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(all_labels, all_preds, target_names=['Class 0', 'Class 1']))
    
    # Matrice de confusion visuelle
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predictions')
    plt.ylabel('Actual')
    plt.show()

def get_class_weights(dataset: Union[ImageFolder, Dataset]) -> torch.Tensor:
    """
    Calcule les poids de classe à partir d'un Dataset PyTorch.
    
    Args:
        dataset: Un objet héritant de torch.utils.data.Dataset (idéalement ImageFolder).
        
    Returns:
        torch.Tensor: Un tenseur contenant les poids pour chaque classe.
    """
    # Vérification de sécurité pour ImageFolder qui possède l'attribut .targets
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
    else:
        # Si c'est un Dataset personnalisé, on extrait les labels manuellement
        # Attention : cela peut être lent sur de très gros datasets
        targets = [label for _, label in dataset]
    
    # Comptage des classes
    # dataset.classes existe sur ImageFolder et contient ['Class_0', 'Class_1']
    num_classes = len(dataset.classes) if hasattr(dataset, 'classes') else len(set(targets))
    
    class_counts = torch.tensor([targets.count(i) for i in range(num_classes)])
    
    total_samples = len(targets)
    
    # Formule : poids = N / (C * n_i)
    weights = total_samples / (num_classes * class_counts.float())
    
    return weights

def show_img_size_and_channels(img_path):
    """
    Affiche la taille et le nombre de canaux d'une image donnée.
    """
    try:
        with Image.open(img_path) as img:
            print(f"Image: {img_path} - Size: {img.size} (Width x Height) - Channels: {len(img.getbands())}")
    except Exception as e:
        print(f"Erreur lors de l'ouverture de l'image {img_path}: {e}")

def plot_history(hist):
    plt.figure(figsize=(12, 4))
    
    # --- Graphique de la Perte (Loss) ---
    plt.subplot(1, 2, 1)
    # On utilise 'loss' au lieu de 'train_loss' pour matcher ton wrapper
    plt.plot(hist['loss'], label='Train', linewidth=2)
    plt.plot(hist['val_loss'], label='Val', linestyle='--')
    plt.title('Loss (Perte)', fontsize=12, fontweight='bold')
    plt.xlabel('Époques')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- Graphique du F1-Score ---
    plt.subplot(1, 2, 2)
    # On utilise 'val_accuracy' pour matcher ton wrapper
    plt.plot(hist['val_accuracy'], label='Val accuracy', color='green', linewidth=2)
    plt.title('F1 Score', fontsize=12, fontweight='bold')
    plt.xlabel('Époques')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    
def display_random_samples(root_path, n_samples=5, max_char=15):
    """
    Affiche une grille d'images aléatoires avec le nom du fichier tronqué au-dessus.
    """
    def _auto_font_size(text, base_size, min_size, target_chars):
        if not text:
            return base_size
        if len(text) <= target_chars:
            return base_size
        scale = target_chars / len(text)
        return max(min_size, base_size * scale)

    # 1. Lister les classes
    classes = sorted([d for d in os.listdir(root_path) 
                      if os.path.isdir(os.path.join(root_path, d)) and not d.startswith('.')])
    
    n_classes = len(classes)
    if n_classes == 0:
        print("Aucun dossier de classe trouvé.")
        return

    # 2. Préparation de la figure
    fig, axes = plt.subplots(n_classes, n_samples, figsize=(n_samples * 3, n_classes * 3))
    
    # Gestion du cas où il n'y a qu'une classe ou un seul échantillon
    if n_classes == 1: axes = axes.reshape(1, -1)
    if n_samples == 1: axes = axes.reshape(-1, 1)

    # 3. Remplissage de la grille
    for i, class_name in enumerate(classes):
        class_dir = os.path.join(root_path, class_name)
        images = []
        for current_root, _, files in os.walk(class_dir):
            for file_name in files:
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    images.append(os.path.join(current_root, file_name))
        
        n_to_show = min(n_samples, len(images))
        selected_images = random.sample(images, n_to_show) if n_to_show > 0 else []
        
        for j in range(n_samples):
            ax = axes[i, j]
            if j < len(selected_images):
                img_path = selected_images[j]
                filename = os.path.basename(img_path)
                
                # Charger et afficher l'image
                img = Image.open(img_path)
                ax.imshow(img)
                
                # --- Gestion du titre (nom du fichier) ---
                # On tronque si c'est trop long : "tres_long_nom_image.jpg" -> "tres_long_nom..."
                display_name = filename if len(filename) <= max_char else filename[:max_char-3] + "..."
                title_fontsize = _auto_font_size(display_name, base_size=9, min_size=6, target_chars=max_char)
                ax.set_title(display_name, fontsize=title_fontsize)
                
                # Nom de la classe sur la gauche de la ligne
                if j == 0:
                    class_fontsize = _auto_font_size(class_name, base_size=12, min_size=8, target_chars=15)
                    ax.set_ylabel(class_name, fontsize=class_fontsize, fontweight='bold', color='red')
            
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.show()

def fix_images(data_dir):

    """
    Check all images in the given directory and delete any that are not of an accepted type.
    """

    IMG_TYPE_ACCEPTED = ["bmp" , "gif", "jpeg", "png"]
    IMG_EXTS = [".png", ".jpg", ".jpeg"]
    for filepath in Path(data_dir).rglob("*"):
        if filepath.suffix.lower() in IMG_EXTS:
            img_type = imghdr.what(filepath)
            if img_type not in IMG_TYPE_ACCEPTED:
                print(f"Deleting file: {filepath} of type: {img_type}")
                os.remove(filepath)
            else: 
                print(f"Accepted file: {filepath} of type: {img_type}")
                
    print("Image fixing completed.")

def get_files_by_class(data_dir):
    """
    Build {class_name: [file_paths]} from a directory, with or without nested folders.

    Supported layouts:
    1) data_dir/class_name/image.jpg
    2) data_dir/class_name/subfolder/.../image.jpg
    3) data_dir/image.jpg  -> class "unlabeled"
    """

    data_dir = Path(data_dir)
    dataset = {}

    for file_path in data_dir.rglob("*"):
        if not file_path.is_file():
            continue

        rel = file_path.relative_to(data_dir)
        class_name = rel.parts[0] if len(rel.parts) > 1 else "unlabeled"

        dataset.setdefault(class_name, []).append(file_path)

    return dataset

def show_class_distribution(data_dir):

    """
    Print the number of samples in each class.
    """

    dataset = get_files_by_class(data_dir)
    for class_name, files in dataset.items():
        print(f"Class: {class_name} - Number of samples: {len(files)}")

def plot_class_distribution(data_dir , title="Class Distribution"):

    """
    Plot class distribution as a bar chart using Plotly.
    """
    dataset = get_files_by_class(data_dir)

    counts = {class_name: len(files) for class_name, files in dataset.items()}
    
    df = pd.DataFrame({
        "Class": list(counts.keys()),
        "Count": list(counts.values())
    })
    
    # Plot
    fig = px.bar(df, x="Class", y="Count", text="Count")
    fig.update_layout(title=title)
    fig.show()

def create_train_test_val_split(data_dir,output_dir="./data/organized", train_ratio=0.7,val_ratio=0.15,test_ratio=0.15,seed=42, allowed_classes=None): 

    """
    Create train, val, test splits from a dataset organized as:
    data_dir/
        class0/
            img1.jpg
            img2.jpg
        class1/
            img3.jpg
            img4.jpg
    """

    random.seed(seed)

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    assert train_ratio + val_ratio + test_ratio == 1.0 # Ensure ratios sum to 1

    dataset = get_files_by_class(data_dir)

    for class_name, images in dataset.items(): 
        if allowed_classes is not None and class_name not in allowed_classes:
            print(f"Skipping class: {class_name} as it's not in allowed_classes")
            continue

        random.shuffle(images)

        n = len(images)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:]
        }

        for split_name, split_files in splits.items(): 
            split_class_dir = output_dir / split_name / class_name
            split_class_dir.mkdir(parents=True, exist_ok=True)

            for img_path in split_files: 
                shutil.copy(img_path, split_class_dir / img_path.name)

    print("Dataset split completed. Organized data saved to:", output_dir)

def sanity_check_labels(dataset, original_data_dir, max_checks=None):
    """
    Verify that each file in the Keras dataset corresponds to the correct original class folder.
    This version dynamically uses class_names from the dataset.
    """

    original_data_dir = Path(original_data_dir)
    file_paths = dataset.file_paths
    class_names = dataset.class_names # Ex: ['AI', 'Real']

    errors = 0
    checked_count = 0

    print(f"Checking against classes: {class_names}")

    for idx, file_path in enumerate(file_paths):
        file_path = Path(file_path)
        filename = file_path.name

        # On récupère le nom de la classe via le dossier parent dans le dataset split
        # Ex: .../organized/train/AI/image.jpg -> parent.name = "AI"
        current_class_name = file_path.parent.name
        
        # On vérifie si ce fichier existe dans le dossier source d'origine
        # On suppose que original_data_dir a la structure : data_dir/AI/image.jpg
        expected_path = original_data_dir / current_class_name / filename

        if not expected_path.exists():
            # Parfois les structures sont data_dir/Class/Subfolder/Image. 
            # On tente une recherche récursive si le chemin direct échoue.
            found = list(original_data_dir.rglob(filename))
            if not found:
                print(f"❌ Mismatch: {filename} (Class: {current_class_name}) not found in original source.")
                errors += 1
            else:
                # Si trouvé ailleurs, on vérifie que c'est bien le bon dossier de classe
                found_parent = found[0].parent.name
                # (Ou un parent plus haut si sous-dossiers)
                if current_class_name not in str(found[0]):
                     print(f"⚠️ Label Mismatch potential: {filename} is labelled {current_class_name} but found in {found[0]}")
                     errors += 1

        checked_count += 1
        if max_checks is not None and checked_count >= max_checks:
            break

    if errors == 0:
        print(f"✅ Sanity check passed. {checked_count} files verified.")
    else:
        print(f"⚠️ Found {errors} mismatches out of {checked_count} checks.")

def create_validation_split(source_dir, val_dir, split_percentage, seed=None):
    """
    Moves a percentage of files from source_dir to val_dir, preserving the inner folder structure.
    
    Args:
        source_dir (str): Path to the source directory (e.g., 'data/train').
        val_dir (str): Path to the validation directory (e.g., 'data/val').
        split_percentage (float): Percentage of files to move (e.g., 0.2 for 20% or 20 for 20%).
        seed (int, optional): Random seed for reproducibility.
    """
    if seed is not None:
        random.seed(seed)

    if split_percentage > 1:
        split_percentage = split_percentage / 100.0

    if not os.path.exists(source_dir):
        print(f"Source directory {source_dir} does not exist.")
        return

    for class_name in os.listdir(source_dir):
        class_source_dir = os.path.join(source_dir, class_name)
        
        if not os.path.isdir(class_source_dir):
            continue
            
        class_val_dir = os.path.join(val_dir, class_name)
        os.makedirs(class_val_dir, exist_ok=True)
        
        files = [f for f in os.listdir(class_source_dir) if os.path.isfile(os.path.join(class_source_dir, f))]
        
        num_files_to_move = int(len(files) * split_percentage)
        files_to_move = random.sample(files, num_files_to_move)
        
        for file_name in files_to_move:
            src_path = os.path.join(class_source_dir, file_name)
            dst_path = os.path.join(class_val_dir, file_name)
            shutil.move(src_path, dst_path)
            
        print(f"Moved {num_files_to_move} files from '{class_source_dir}' to '{class_val_dir}'")

def check_duplicates(*directories):
    """
    Vérifie s'il y a des doublons (fichiers avec le même contenu exact) 
    dans et entre les dossiers spécifiés (ex: train, test, val).
    Utilise le hachage MD5 pour comparer le contenu des images.
    
    Args:
        *directories: Chemins des dossiers à vérifier (ex: 'data/train', 'data/test', 'data/val').
    """
    seen_hashes = {}
    duplicates = []
    
    for directory in directories:
        if not os.path.exists(directory):
            print(f"⚠️ Le dossier {directory} n'existe pas.")
            continue
            
        for root, _, files in os.walk(directory):
            for file in files:
                # On ignore les fichiers cachés
                if file.startswith('.'):
                    continue
                    
                file_path = os.path.join(root, file)
                
                # Calcul du hash MD5 du fichier
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                
                if file_hash in seen_hashes:
                    duplicates.append((file, seen_hashes[file_hash], file_path))
                else:
                    seen_hashes[file_hash] = file_path
                    
    if duplicates:
        print(f"⚠️ {len(duplicates)} doublons trouvés (contenu identique) !")
        print("Exemples de doublons :")
        for dup in duplicates[:5]:
            print(f" - Fichier : {dup[0]}")
            print(f"   1. {dup[1]}")
            print(f"   2. {dup[2]}")
        if len(duplicates) > 5:
            print("   ...")
    else:
        print("✅ Aucun doublon de contenu trouvé dans ou entre les dossiers.")
        
    return duplicates

def remove_duplicates(*directories):
    """
    Trouve et supprime les doublons (fichiers avec le même contenu exact) 
    dans et entre les dossiers spécifiés.
    
    Args:
        *directories: Chemins des dossiers à vérifier (ex: 'data/train', 'data/test', 'data/val').
    """
    print("Recherche des doublons à supprimer...")
    duplicates = check_duplicates(*directories)
    
    if not duplicates:
        print("Aucun doublon à supprimer.")
        return 0
        
    removed_count = 0
    for dup in duplicates:
        duplicate_path = dup[2]
        try:
            if os.path.exists(duplicate_path):
                os.remove(duplicate_path)
                removed_count += 1
        except Exception as e:
            print(f"Erreur lors de la suppression de {duplicate_path}: {e}")
            
    print(f"✅ {removed_count} doublons supprimés avec succès.")
    return removed_count

def create_spatial_train_test_val_split(data_dir, output_dir="./data/organized_spatial", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42, allowed_classes=None): 
    """
    Crée un split Train/Val/Test garanti SANS fuite spatiale.
    Groupe les données par région géographique avant de les séparer.
    """
    print("🌍 Démarrage du split spatial (Group-based)...")
    random.seed(seed)
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    # Tolérance pour les flottants (ex: 0.8 + 0.1 + 0.1 = 1.0)
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Les ratios doivent faire 1.0"

    dataset = get_files_by_class(data_dir)

    # ---------------------------------------------------------
    # ÉTAPE 1 : Identifier toutes les régions uniques au global
    # ---------------------------------------------------------
    all_regions = set()
    for class_name, images in dataset.items(): 
        if allowed_classes is not None and class_name not in allowed_classes:
            continue
        for img_path in images:
            all_regions.add(extract_scene_id(img_path.stem)) # .stem enlève le .jpg
            
    unique_regions = list(all_regions)
    unique_regions.sort() # On trie d'abord pour garantir que la seed aura toujours le même effet
    random.shuffle(unique_regions)

    print(f"🗺️ Nombre total de régions uniques trouvées : {len(unique_regions)}")
    print(f"Exemples de régions : {unique_regions[:5]}")

    # ---------------------------------------------------------
    # ÉTAPE 2 : Découper les régions (et non les images)
    # ---------------------------------------------------------
    n_regions = len(unique_regions)
    train_end = int(n_regions * train_ratio)
    val_end = int(n_regions * (train_ratio + val_ratio))

    # On transforme en 'set' pour que la recherche (if region in train_regions) soit ultra rapide
    train_regions = set(unique_regions[:train_end])
    val_regions = set(unique_regions[train_end:val_end])
    test_regions = set(unique_regions[val_end:])
    
    print(f"📊 Répartition des régions -> Train: {len(train_regions)} | Val: {len(val_regions)} | Test: {len(test_regions)}")

    # ---------------------------------------------------------
    # ÉTAPE 3 : Copier les fichiers dans les bons dossiers
    # ---------------------------------------------------------
    for class_name, images in dataset.items(): 
        if allowed_classes is not None and class_name not in allowed_classes:
            continue

        for img_path in images: 
            region = extract_scene_id(img_path.stem)
            
            # On détermine la destination de l'image en fonction de sa région
            if region in train_regions:
                split_name = "train"
            elif region in val_regions:
                split_name = "val"
            else:
                split_name = "test"
                
            split_class_dir = output_dir / split_name / class_name
            split_class_dir.mkdir(parents=True, exist_ok=True)

            shutil.copy(img_path, split_class_dir / img_path.name)

    print(f"✅ Split spatial terminé ! Données étanches sauvegardées dans : {output_dir}")

