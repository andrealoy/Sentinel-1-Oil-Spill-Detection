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
from pathlib import Path

# Data manipulation and numerical arrays
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

# Machine Learning and Deep Learning
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    precision_score, 
    recall_score, 
    f1_score
)

def plot_training_history(history):
    # Récupération des données
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(15, 5))

    # --- Graphique de l'Accuracy ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Entraînement', linewidth=2)
    plt.plot(epochs_range, val_acc, label='Validation', linestyle='--')
    plt.title('Précision (Accuracy)', fontsize=14, fontweight='bold')
    plt.xlabel('Époques')
    plt.ylabel('Score')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    # --- Graphique de la Perte ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Entraînement', linewidth=2)
    plt.plot(epochs_range, val_loss, label='Validation', linestyle='--')
    plt.title('Perte (Loss)', fontsize=14, fontweight='bold')
    plt.xlabel('Époques')
    plt.ylabel('Erreur')
    plt.legend(loc='upper right')
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


def evaluate_model_on_folder(
    model,
    folder_path,
    image_size=(512, 512),
    batch_size=32,
    seed=42,
    n_display=0,
):
    """
    Evaluate a trained Keras model on a folder structured as:
    folder_path/
        class_a/
        class_b/
        ...

    Class names are inferred automatically from subfolders (no hardcoding).
    If n_display > 0, show n validation images with predictions.
    Returns a dict with metrics and labels.
    """

    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    class_dirs = sorted([d.name for d in folder_path.iterdir() if d.is_dir() and not d.name.startswith(".")])
    if not class_dirs:
        raise ValueError(f"No class subfolders found in: {folder_path}")

    val_ds = tf.keras.utils.image_dataset_from_directory(
        folder_path,
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
        shuffle=False,
        color_mode="rgb",
    )

    y_true = np.concatenate([labels.numpy() for _, labels in val_ds], axis=0)
    raw_preds = model.predict(val_ds, verbose=0)

    if raw_preds.ndim == 1 or (raw_preds.ndim == 2 and raw_preds.shape[1] == 1):
        y_pred = (np.ravel(raw_preds) >= 0.5).astype(int)
    else:
        y_pred = np.argmax(raw_preds, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=val_ds.class_names,
        output_dict=True,
        zero_division=0,
    )

    class_names = val_ds.class_names
    print("\n=== Validation results ===")
    print(f"Class 0 = {class_names[0]}")
    if len(class_names) > 1:
        print(f"Class 1 = {class_names[1]}")
    print(f"Classes detected: {class_names}")
    print(f"Precision (macro): {precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Recall (macro):    {recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"F1-score (macro):  {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    print("Confusion matrix:")
    print(cm)

    if n_display and n_display > 0:
        file_paths = [Path(p) for p in val_ds.file_paths]
        n_display = min(n_display, len(file_paths))

        rng = np.random.default_rng(seed)
        selected_indices = rng.choice(len(file_paths), size=n_display, replace=False)

        cols = min(4, n_display)
        rows = int(np.ceil(n_display / cols))
        plt.figure(figsize=(4 * cols, 4 * rows))

        for plot_pos, sample_idx in enumerate(selected_indices, start=1):
            img_path = file_paths[sample_idx]

            img = tf.keras.utils.load_img(img_path, target_size=image_size)
            img_arr = tf.keras.utils.img_to_array(img)
            input_tensor = np.expand_dims(img_arr, axis=0)

            sample_pred = model.predict(input_tensor, verbose=0)
            if sample_pred.ndim == 1 or (sample_pred.ndim == 2 and sample_pred.shape[1] == 1):
                pred_idx = int(np.ravel(sample_pred)[0] >= 0.5)
                confidence = float(np.ravel(sample_pred)[0])
            else:
                pred_idx = int(np.argmax(sample_pred, axis=1)[0])
                confidence = float(np.max(sample_pred, axis=1)[0])

            true_idx = int(y_true[sample_idx])
            true_label = class_names[true_idx]
            pred_label = class_names[pred_idx]

            ax = plt.subplot(rows, cols, plot_pos)
            ax.imshow(img)
            ax.set_title(f"True: {true_label}\nPred: {pred_label} ({confidence:.2f})")
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    return {
        "class_names": class_names,
        "y_true": y_true,
        "y_pred": y_pred,
        "confusion_matrix": cm,
        "report": report_dict,
    }

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



