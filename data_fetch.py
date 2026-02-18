import kaggle
import os 

# Download latest version
kaggle.api.authenticate()
path = './data'

download = kaggle.api.dataset_download_files("harikrishnacs/sentinel-1-sar-oil-spill-detection-dataset",path=path,unzip=True)
print("="*50)
print("Path to dataset files:", path)
print("="*50)
print("Folders downloaded:")
print(os.listdir(path))
print("="*50)
print("Absolute path:", os.path.abspath(path))
print("="*50)