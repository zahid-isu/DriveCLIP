import clip
import torch
import torch.utils.data as data
from PIL import Image
from pathlib import Path
import json
import joblib
from torch.utils.data import DataLoader
from tqdm import tqdm
from inference.DriveCLIP.inference.frame import extract_frames

# Select model backbone
model_name= 'ViT-L/14'

new_data_dir = f"frame" # replace with your new data directory
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(model_name, device) # load CLIP model backbone (ViT encoders)

# Extract frame from video and create dataset
extract_frames('video/Dashboard_user_id_13522_NoAudio_5.MP4', 'frame', fps=1)  # Change fps if needed
class UnlabeledImageDataset(data.Dataset):
    def __init__(self, folder, transform=None):
        self.transform = transform
        self.image_files = list(Path(folder).glob('*.jpg'))  # adjust if you have a different file type

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, str(image_path)  # convert the path to a string

def get_features(dataset):
    all_features = []
    filepaths = []
    print("running inference...")

    with torch.no_grad():
        for images, paths in tqdm(DataLoader(dataset, batch_size=1, shuffle=False)):
            images = images.to(device)
            features = model.encode_image(images)
            all_features.append(features.cpu())
            filepaths.extend(paths)

    return torch.cat(all_features).numpy(), filepaths 

# Running the inference on the feature map
new_dataset = UnlabeledImageDataset(new_data_dir, transform=preprocess)
new_features, new_filepaths = get_features(new_dataset)

# Loading trained-classifier weights
classifier = joblib.load(f"model/syn2fps_models/ViT-L14/vitl14-per1_429_1000.pkl") # Change the model path

new_predictions = classifier.predict(new_features)
new_probs = classifier.predict_proba(new_features)

results = {}
for i in range(len(new_filepaths)):
    results[new_filepaths[i]] = {
        'prediction': int(new_predictions[i]),
        'prob_score': new_probs[i].tolist() 
    }

with open('frame_predictions.json', 'w') as f:
    json.dump(results, f)

print("Model finished")