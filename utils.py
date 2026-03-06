#######################################################################################################
#                                                                                                     #
# A collection of utility functions for image dataset management, visualization, and model evaluation.#
# Authors: Rose Aupepin, Andréa Loy, Alizée Robin, Omar Zeroual                                       #  
#                                                                                                     #
#######################################################################################################

# 1. Standard Library Imports
import os
import random
import shutil
import hashlib
import imghdr
from pathlib import Path
from collections import Counter
from typing import Union

# 2. Third-party Libraries
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
from tqdm import tqdm

# 3. Deep Learning (PyTorch & Tools)
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# 4. Machine Learning & Metrics
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    precision_score, 
    recall_score, 
    f1_score
)
from sklearn.utils.class_weight import compute_class_weight

# --- Dataset Management & Cleaning ---

def clean_low_variance_dataset(root_dir, threshold=2.0):
    """
    Removes images with very low variance (near-solid colors or empty tiles).
    """
    removed_count = 0
    for subdir, _, files in os.walk(root_dir):
        for file in tqdm(files, desc=f"Cleaning {os.path.basename(subdir)}"):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                path = os.path.join(subdir, file)
                # Load image in grayscale
                img = np.array(Image.open(path).convert('L'))
                
                # Delete if standard deviation is below threshold
                if img.std() < threshold:
                    os.remove(path)
                    removed_count += 1
                    
    print(f"✅ Cleaning complete: {removed_count} useless images removed.")

def find_and_remove_black_images(data_dir, threshold=1.0, delete=False):
    """
    Scans directory for near-black images based on pixel standard deviation.
    """
    data_dir = Path(data_dir)
    black_images = []
    
    for img_path in data_dir.rglob("*.jpg"):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            if np.std(img) < threshold:
                black_images.append(img_path)
                
    print(f"🧹 Detected black images: {len(black_images)}")
    
    if delete:
        for img_path in black_images:
            os.remove(img_path)
        print("🗑️ All black images have been deleted!")
        
    return black_images

def show_class_distribution(data_dir):
    """
    Prints the number of samples in each class to the console.
    """
    dataset = get_files_by_class(data_dir)
    for class_name, files in dataset.items():
        print(f"Class: {class_name} - Number of samples: {len(files)}")

def plot_class_distribution(data_dir, title="Class Distribution"):
    """
    Plots the class distribution as a bar chart using Plotly for interactive visualization.
    """
    dataset = get_files_by_class(data_dir)

    # Prepare data for plotting
    counts = {class_name: len(files) for class_name, files in dataset.items()}
    
    df = pd.DataFrame({
        "Class": list(counts.keys()),
        "Count": list(counts.values())
    })
    
    # Create interactive bar chart
    fig = px.bar(
        df, 
        x="Class", 
        y="Count", 
        text="Count",
        color="Class",
        template="plotly_white"
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Category",
        yaxis_title="Number of Images",
        showlegend=False
    )
    fig.show()

# --- SAR Image Preprocessing ---

def preprocess_sar(image_np):
    """
    Specific SAR preprocessing: clipping outliers, scaling, and median filtering.
    """
    # 1. Clip outliers (pixels saturated at 255/artifacts)
    # Cap at 99th percentile to bring burned areas back into useful range
    p99 = np.percentile(image_np, 99)
    image_clipped = np.clip(image_np, 0, p99)
    
    # 2. Scale normalization for OpenCV (back to uint8)
    # Scale so p99 becomes 255 to maintain dynamic range
    denominator = image_clipped.max() - image_clipped.min()
    if denominator == 0:
        return image_np.astype(np.uint8)
        
    image_scaled = ((image_clipped - image_clipped.min()) * (255 / denominator)).astype(np.uint8)
    
    # 3. Median Denoising (to smooth speckle noise while preserving edges)
    denoised = cv2.medianBlur(image_scaled, 3)
    
    return denoised

def denoise_sar(image_np):
    """Simple median blur for speckle reduction."""
    return cv2.medianBlur(image_np, 3)

# --- Data Splitting & Analysis ---

def create_spatial_split_full(data_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Splits data into train/val/test based on geographic regions to avoid spatial leakage.
    """
    random.seed(seed)
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    # 1. Analyze regions
    region_map = {} 
    all_images_count = 0
    
    print(f"🔍 Analyzing source folder: {data_dir}")
    
    for img_path in data_dir.rglob('*.jpg'): 
        if img_path.is_file() and not img_path.name.startswith('.'):
            region = extract_scene_id(img_path.stem)
            if region not in region_map:
                region_map[region] = []
            region_map[region].append(img_path)
            all_images_count += 1

    if all_images_count == 0:
        print("❌ NO IMAGES FOUND! Check extension or path.")
        return

    # 2. Sorting and Distribution
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

    print(f"\n📊 Final Distribution: {train_count} Train | {val_count} Val | {test_count} Test")

    # 3. Copying files
    print("🚀 Copying files...")
    for split_name, regions in [('train', train_regions), ('val', val_regions), ('test', test_regions)]:
        for r in regions:
            for img_path in region_map[r]:
                class_name = img_path.parent.name 
                dest_dir = output_dir / split_name / class_name
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_path, dest_dir / img_path.name)

    print(f"✅ Success! {all_images_count} images processed.")

def count_images_by_region(data_dir):
    """
    Counts images per region (e.g., VEN, GBR, BAH) using geo-codes in filenames.
    """
    data_dir = Path(data_dir)
    region_counts = Counter()

    for img_path in data_dir.rglob("*.jpg"):
        scene_id = extract_scene_id(img_path.stem)
        region_counts[scene_id] += 1

    print("Image count per region:")
    for region, count in region_counts.items():
        print(f"{region}: {count} images.")

def extract_scene_id(filename_stem):
    """
    Extracts the geo-code (VEN, GBR, BAH) from filename stem.
    Expected format: '0_0_0_img_ID_REGION_cls_0'
    """
    parts = filename_stem.split('_')
    if len(parts) >= 3:
        # Region code is usually the 3rd to last block
        return parts[-3] 
    else:
        return "UNKNOWN"

# --- Visualization & Explainability ---

def verify_batch(loader, std, mean):
    """Displays the first batch of images after de-normalization."""
    batch_images, batch_labels = next(iter(loader))
    plt.figure(figsize=(10, 5))
    for i in range(4):
        plt.subplot(1, 4, i+1)
        # De-normalize for display
        img = batch_images[i].squeeze().numpy()
        img = (img * std[0]) + mean[0] 
        plt.imshow(img, cmap='gray')
        plt.title(f"Label: {batch_labels[i]}")
        plt.axis('off')
    plt.show()

def visualize_random_gradcam(model, target_layer, dataset, device, num_images=5):
    """
    Visualizes Grad-CAM activation maps for random images in the dataset.
    """
    model.eval()
    cam = GradCAM(model=model, target_layers=[target_layer])
    indices = random.sample(range(len(dataset)), num_images)
    
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 4 * num_images))
    if num_images == 1:
        axes = [axes]
        
    for i, idx in enumerate(indices):
        img_tensor, true_label = dataset[idx]
        input_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Generate Grad-CAM mask
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        
        # Get prediction for color coding
        with torch.no_grad():
            output = model(input_tensor)
            pred_class = torch.argmax(output, dim=1).item()
            
        color = "green" if pred_class == true_label else "red"
        
        # Prepare original image for display (de-normalize)
        orig_img = img_tensor.squeeze().cpu().numpy()
        img_rgb = np.stack((orig_img,)*3, axis=-1)
        img_rgb = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min() + 1e-8)
        
        visualization = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)
        
        axes[i][0].imshow(orig_img, cmap='gray')
        axes[i][0].set_title(f"Original (True Class: {true_label})")
        axes[i][0].axis('off')
        
        axes[i][1].imshow(visualization)
        axes[i][1].set_title(f"Grad-CAM (Pred: {pred_class})", color=color, fontweight='bold')
        axes[i][1].axis('off')
        
    plt.tight_layout()
    plt.show()

def plot_pytorch_history(history):
    """Plots training/validation loss and F1-score curves."""
    epochs_range = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(15, 5))

    # --- Loss Plot ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Training', color='#1f77b4', linewidth=2)
    plt.plot(epochs_range, history['val_loss'], label='Validation', color='#ff7f0e', linestyle='--')
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- F1-Score Plot ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['val_f1'], label='Validation F1', color='#2ca02c', linewidth=2)
    plt.title('Validation F1 Score', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# --- Helpers & Stats ---

def set_seed(seed=42):
    """Sets all random seeds for reproducible training."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed} 🌱")

def get_mean_and_std(loader):
    """Calculates mean and standard deviation of the dataset for normalization."""
    mean = 0.
    std = 0.
    total_images_count = 0
    
    for images, _ in loader:
        batch_samples = images.size(0) 
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count
    
    return mean, std

# --- Model Evaluation & Metrics ---

def evaluate_test_set(model, test_loader, device):
    """
    Evaluates the model on the test set and displays classification metrics and confusion matrix.
    """
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
    
    # Display full report (Precision, Recall, F1-score)
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(all_labels, all_preds, target_names=['Class 0', 'Class 1']))
    
    # Visual Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predictions')
    plt.ylabel('Actual')
    plt.show()

# --- Class Balancing ---

def get_class_weights(dataset):
    """
    Calculates balanced class weights to handle dataset imbalance.
    Formula: weight = n_samples / (n_classes * n_samples_at_class)
    """
    targets = dataset.targets
    classes = np.unique(targets)
    
    weights = compute_class_weight(
        class_weight='balanced', 
        classes=classes, 
        y=targets
    )
    
    return torch.tensor(weights, dtype=torch.float)

def create_balanced_sampler(dataset):
    """
    Creates a WeightedRandomSampler to ensure each batch has an equal distribution of classes.
    """
    targets = np.array(dataset.targets)
    class_counts = np.bincount(targets)

    # Calculate weights per class (inverse frequency)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[targets]

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler, class_counts.tolist()

# --- Image Inspection & Visualization ---

def show_img_size_and_channels(img_path):
    """Displays the dimensions and channel count of a specific image."""
    try:
        with Image.open(img_path) as img:
            print(f"Image: {img_path} - Size: {img.size} (WxH) - Channels: {len(img.getbands())}")
    except Exception as e:
        print(f"Error opening image {img_path}: {e}")

def plot_history(hist):
    """Plots training/validation Loss, Accuracy, and F1-Score history."""
    has_val_f1 = 'val_f1' in hist and len(hist['val_f1']) > 0
    n_cols = 3 if has_val_f1 else 2
    plt.figure(figsize=(6 * n_cols, 4))

    # --- Loss Plot ---
    plt.subplot(1, n_cols, 1)
    plt.plot(hist['loss'], label='Train loss', linewidth=2)
    plt.plot(hist['val_loss'], label='Val loss', linestyle='--')
    plt.title('Loss', fontsize=12, fontweight='bold')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- Accuracy Plot ---
    plt.subplot(1, n_cols, 2)
    plt.plot(hist['accuracy'], label='Train acc', linewidth=2)
    plt.plot(hist['val_accuracy'], label='Val acc', linestyle='--')
    plt.title('Accuracy', fontsize=12, fontweight='bold')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- Validation F1-Score Plot ---
    if has_val_f1:
        plt.subplot(1, n_cols, 3)
        plt.plot(hist['val_f1'], label='Val F1', color='green', linewidth=2)
        plt.title('Validation F1', fontsize=12, fontweight='bold')
        plt.xlabel('Epochs')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def display_random_samples(root_path, n_samples=5, max_char=15):
    """Displays a grid of random images with truncated filenames above them."""
    def _auto_font_size(text, base_size, min_size, target_chars):
        if not text or len(text) <= target_chars:
            return base_size
        scale = target_chars / len(text)
        return max(min_size, base_size * scale)

    # 1. List classes
    classes = sorted([d for d in os.listdir(root_path) 
                      if os.path.isdir(os.path.join(root_path, d)) and not d.startswith('.')])
    
    n_classes = len(classes)
    if n_classes == 0:
        print("No class folders found.")
        return

    fig, axes = plt.subplots(n_classes, n_samples, figsize=(n_samples * 3, n_classes * 3))
    
    if n_classes == 1: axes = axes.reshape(1, -1)
    if n_samples == 1: axes = axes.reshape(-1, 1)

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
                
                img = Image.open(img_path)
                ax.imshow(img)
                
                # Truncate filename if too long
                display_name = filename if len(filename) <= max_char else filename[:max_char-3] + "..."
                title_fontsize = _auto_font_size(display_name, base_size=9, min_size=6, target_chars=max_char)
                ax.set_title(display_name, fontsize=title_fontsize)
                
                if j == 0:
                    class_fontsize = _auto_font_size(class_name, base_size=12, min_size=8, target_chars=15)
                    ax.set_ylabel(class_name, fontsize=class_fontsize, fontweight='bold', color='red')
            
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.show()

# --- Dataset Integrity & Cleaning ---

def fix_images(data_dir, verbose=False):
    """
    Deletes files with accepted extensions but invalid internal formats.
    """
    IMG_TYPE_ACCEPTED = ["bmp" , "gif", "jpeg", "png"]
    IMG_EXTS = [".png", ".jpg", ".jpeg"]
    for filepath in Path(data_dir).rglob("*"):
        if filepath.suffix.lower() in IMG_EXTS:
            img_type = imghdr.what(filepath)
            if img_type not in IMG_TYPE_ACCEPTED:
                if verbose:
                    print(f"Deleting file: {filepath} (Invalid type: {img_type})")
                os.remove(filepath)
            else: 
                if verbose:
                    print(f"Accepted file: {filepath} (Type: {img_type})")
                
    print("Image fixing completed.")

def check_duplicates(*directories):
    """
    Checks for duplicate files (exact same content) within and across specified directories using MD5 hashing.
    """
    seen_hashes = {}
    duplicates = []
    
    for directory in directories:
        if not os.path.exists(directory):
            print(f"⚠️ Directory {directory} does not exist.")
            continue
            
        for root, _, files in os.walk(directory):
            for file in files:
                if file.startswith('.'): continue
                file_path = os.path.join(root, file)
                
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                
                if file_hash in seen_hashes:
                    duplicates.append((file, seen_hashes[file_hash], file_path))
                else:
                    seen_hashes[file_hash] = file_path
                    
    if duplicates:
        print(f"⚠️ {len(duplicates)} duplicates found (identical content)!")
        for dup in duplicates[:5]:
            print(f" - File: {dup[0]}\n   1. {dup[1]}\n   2. {dup[2]}")
    else:
        print("✅ No duplicate content found across directories.")
        
    return duplicates

def remove_duplicates(*directories):
    """Finds and deletes exact duplicate files."""
    print("Searching for duplicates to remove...")
    duplicates = check_duplicates(*directories)
    
    if not duplicates:
        print("No duplicates to remove.")
        return 0
        
    removed_count = 0
    for dup in duplicates:
        duplicate_path = dup[2]
        try:
            if os.path.exists(duplicate_path):
                os.remove(duplicate_path)
                removed_count += 1
        except Exception as e:
            print(f"Error deleting {duplicate_path}: {e}")
            
    print(f"✅ {removed_count} duplicates successfully removed.")
    return removed_count

# --- Dataset Splitting (Standard & Spatial) ---

def create_train_test_val_split(data_dir, output_dir="./data/organized", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42, allowed_classes=None): 
    """Standard random split for datasets organized by class folders."""
    random.seed(seed)
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5

    dataset = get_files_by_class(data_dir)
    for class_name, images in dataset.items(): 
        if allowed_classes and class_name not in allowed_classes: continue
        random.shuffle(images)
        n = len(images)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        splits = {"train": images[:train_end], "val": images[train_end:val_end], "test": images[val_end:]}
        for split_name, split_files in splits.items(): 
            split_class_dir = output_dir / split_name / class_name
            split_class_dir.mkdir(parents=True, exist_ok=True)
            for img_path in split_files:
                shutil.copy(img_path, split_class_dir / img_path.name)
    print("Dataset split completed. Output saved to:", output_dir)

def create_spatial_train_test_val_split(data_dir, output_dir="./data/organized_spatial", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42, allowed_classes=None): 
    """
    Creates a Train/Val/Test split guaranteed to have NO spatial leakage.
    Groups data by geographic region before splitting.
    """
    print("🌍 Starting Spatial Split (Group-based)...")
    random.seed(seed)
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5

    dataset = get_files_by_class(data_dir)
    all_regions = set()
    for class_name, images in dataset.items(): 
        if allowed_classes and class_name not in allowed_classes: continue
        for img_path in images:
            all_regions.add(extract_scene_id(img_path.stem))
            
    unique_regions = sorted(list(all_regions))
    random.shuffle(unique_regions)

    n_regions = len(unique_regions)
    train_end = int(n_regions * train_ratio)
    val_end = int(n_regions * (train_ratio + val_ratio))

    train_regions = set(unique_regions[:train_end])
    val_regions = set(unique_regions[train_end:val_end])
    test_regions = set(unique_regions[val_end:])
    
    print(f"📊 Region Distribution -> Train: {len(train_regions)} | Val: {len(val_regions)} | Test: {len(test_regions)}")

    for class_name, images in dataset.items(): 
        if allowed_classes and class_name not in allowed_classes: continue
        for img_path in images: 
            region = extract_scene_id(img_path.stem)
            split_name = "train" if region in train_regions else "val" if region in val_regions else "test"
            dest = output_dir / split_name / class_name
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy(img_path, dest / img_path.name)
    print(f"✅ Spatial split complete! Leakage-free data saved at: {output_dir}")

# --- Utility Helpers ---

def get_files_by_class(data_dir):
    """Maps class names to lists of file paths."""
    data_dir = Path(data_dir)
    dataset = {}
    for file_path in data_dir.rglob("*"):
        if not file_path.is_file(): continue
        rel = file_path.relative_to(data_dir)
        class_name = rel.parts[0] if len(rel.parts) > 1 else "unlabeled"
        dataset.setdefault(class_name, []).append(file_path)
    return dataset

def sanity_check_labels(dataset, original_data_dir, max_checks=None):
    """Verifies that split dataset files match their original source class folders."""
    original_data_dir = Path(original_data_dir)
    file_paths = dataset.file_paths
    class_names = dataset.class_names
    errors, checked_count = 0, 0

    print(f"Checking against classes: {class_names}")
    for idx, file_path in enumerate(file_paths):
        file_path = Path(file_path)
        filename, current_class_name = file_path.name, file_path.parent.name
        expected_path = original_data_dir / current_class_name / filename

        if not expected_path.exists():
            found = list(original_data_dir.rglob(filename))
            if not found:
                print(f"❌ Mismatch: {filename} not found in source.")
                errors += 1
            elif current_class_name not in str(found[0]):
                print(f"⚠️ Label potential Mismatch: {filename} labeled {current_class_name} but found in {found[0]}")
                errors += 1

        checked_count += 1
        if max_checks and checked_count >= max_checks: break

    print(f"✅ Check passed ({checked_count} verified)" if errors == 0 else f"⚠️ Found {errors} mismatches.")