import kagglehub

# Download latest version
path = kagglehub.dataset_download("coco/")

print("Path to dataset files:", path)
