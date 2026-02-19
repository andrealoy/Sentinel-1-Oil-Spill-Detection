import matplotlib.pyplot as plt
import imghdr
import os 
import plotly.express as px
import pandas as pd
import random
import shutil
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report , precision_score, recall_score, f1_score
from pathlib import Path

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
    Given a directory structured as:
    data_dir/
        class0/
            img1.jpg
            img2.jpg
        class1/
            img3.jpg
            img4.jpg
    Returns a dict {class_name: [list of file paths]}   
    """

    data_dir = Path(data_dir)
    dataset = {}
    
    for class_dir in data_dir.iterdir():
        if class_dir.is_dir():
            files = list(class_dir.rglob("*.*"))
            dataset[class_dir.name] = files 

    return dataset

def show_class_distribution(data_dir):

    """
    Print the number of samples in each class.
    """

    dataset = get_files_by_class(data_dir)
    for class_name, files in dataset.items():
        print(f"Class: {class_name} - Number of samples: {len(files)}")

def plot_class_distribution(data_dir):

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
    fig.update_layout(title="Class Distribution")
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

def compute_class_weights_from_directory(train_dir):

    """
    Compute class weights from a train directory structured as:
    
    train/
        class0/
        class1/
    
    Returns a dict {class_index: weight}
    """

    train_dir = Path(train_dir)

    class_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    
    class_counts = []
    for class_dir in class_dirs:
        count = len(list(class_dir.rglob("*.*")))
        class_counts.append(count)

    total = sum(class_counts)
    num_classes = len(class_counts)

    class_weights = {
        i: total / (num_classes * count)
        for i, count in enumerate(class_counts)
    }

    return class_weights

def evaluate_binary_model(model, dataset, threshold=0.5):

    y_true = []
    y_pred = []

    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        preds_binary = (preds > threshold).astype(int).flatten()

        y_pred.extend(preds_binary)
        y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("True distribution:", np.bincount(y_true))
    print("Predicted distribution:", np.bincount(y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    return y_true, y_pred


def show_samples_per_class(dataset, n_per_class=6):

    class_names = dataset.class_names
    file_paths = dataset.file_paths

    images_class0 = []
    images_class1 = []

    filenames_class0 = []
    filenames_class1 = []

    idx = 0  # index global pour file_paths

    for images, labels in dataset:
        batch_size = images.shape[0]

        for i in range(batch_size):
            label = labels[i].numpy()
            img = images[i].numpy()
            filename = Path(file_paths[idx]).name

            if label == 0 and len(images_class0) < n_per_class:
                images_class0.append(img)
                filenames_class0.append(filename)

            elif label == 1 and len(images_class1) < n_per_class:
                images_class1.append(img)
                filenames_class1.append(filename)

            idx += 1

            if len(images_class0) >= n_per_class and len(images_class1) >= n_per_class:
                break

        if len(images_class0) >= n_per_class and len(images_class1) >= n_per_class:
            break

    # 🔥 Canvas plus grand
    fig, axes = plt.subplots(2, n_per_class, figsize=(4*n_per_class, 8))

    for i in range(n_per_class):

        axes[0, i].imshow(images_class0[i].astype("uint8"))
        axes[0, i].set_title(f"Class 0\n{filenames_class0[i]}", fontsize=9)
        axes[0, i].axis("off")

        axes[1, i].imshow(images_class1[i].astype("uint8"))
        axes[1, i].set_title(f"Class 1\n{filenames_class1[i]}", fontsize=9)
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    plt.show()

def sanity_check_labels(dataset, original_data_dir, max_checks=None):
    """
    Verify that each file in the Keras dataset corresponds to the correct original class folder.
    """

    original_data_dir = Path(original_data_dir)
    file_paths = dataset.file_paths
    class_names = dataset.class_names

    errors = 0

    for idx, file_path in enumerate(file_paths):

        file_path = Path(file_path)
        filename = file_path.name

        # Récupérer label via nom du dossier organisé
        class_name = file_path.parent.name
        label = class_names.index(class_name)

        if label == 0:
            expected_dir = original_data_dir / "S1SAR_UnBalanced_400by400_Class_0" / "0"
        else:
            expected_dir = original_data_dir / "S1SAR_UnBalanced_400by400_Class_1" / "1"

        expected_path = expected_dir / filename

        if not expected_path.exists():
            print(f"❌ Mismatch: {filename} not found in expected folder {expected_dir}")
            errors += 1

        if max_checks is not None and idx >= max_checks:
            break

    if errors == 0:
        print("✅ Sanity check passed. All files match expected class folders.")
    else:
        print(f"⚠️ Found {errors} mismatches.")


def scan_thresholds(y_true, y_proba, step=0.05):

    results = []

    print("threshold | precision | recall | f1")
    print("------------------------------------")

    for t in np.arange(0.1, 0.9, step):
        y_pred = (y_proba >= t).astype(int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        print(f"{t:8.2f} | {precision:9.3f} | {recall:6.3f} | {f1:6.3f}")

        results.append((t, precision, recall, f1))

    return results

def get_true_and_proba(model, dataset):
    y_true = []
    y_proba = []

    for images, labels in dataset:
        probs = model.predict(images, verbose=0).flatten()
        y_proba.extend(probs)
        y_true.extend(labels.numpy())

    return np.array(y_true), np.array(y_proba)