import argparse
import clip
import torch
import torch.utils.data as data
from PIL import Image
from pathlib import Path
import json
import joblib
from torch.utils.data import DataLoader
from tqdm import tqdm
from frame import extract_frames

def parse_args():
    parser = argparse.ArgumentParser(description='Script to extract frames and run inference')
    parser.add_argument('--video', required=True, help='Path to input video')
    parser.add_argument('--frame', required=True, help='Path to store extracted frames')
    args = parser.parse_args()
    return args

def main():
    # Parse command line arguments
    args = parse_args()

    # Select model backbone
    model_name= 'ViT-L/14'

    new_data_dir = args.frame
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device)

    # Extract frame from video and create dataset
    extract_frames(args.video, args.frame, fps=1)

    class UnlabeledImageDataset(data.Dataset):
        def __init__(self, folder, transform=None):
            self.transform = transform
            self.image_files = list(Path(folder).glob('*.jpg'))

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, idx):
            image_path = self.image_files[idx]
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, str(image_path)

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

    new_dataset = UnlabeledImageDataset(new_data_dir, transform=preprocess)
    new_features, new_filepaths = get_features(new_dataset)

    classifier = joblib.load(f"model_ckpt/vitl14-per1_429_1000.pkl")

    new_predictions = classifier.predict(new_features)
    new_probs = classifier.predict_proba(new_features)

    results = {}
    for i in range(len(new_filepaths)):
        results[new_filepaths[i]] = {
            'prediction': int(new_predictions[i]),
            'prob_score': new_probs[i].tolist() 
        }

    with open('results/frame_predictions.json', 'w') as f:
        json.dump(results, f)

    print("Model finished")

if __name__ == "__main__":
    main()
