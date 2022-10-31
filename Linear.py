import os
import clip
import torch

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-L/14', device)

# Load the dataset
# data_dir =
train = datasets.ImageFolder( 'new_data/new_folder/train/', transform=preprocess)
val = datasets.ImageFolder( 'new_data/zero/val/', transform=preprocess)
test = datasets.ImageFolder( 'new_data/new_folder/test/', transform=preprocess)

print(len(train), len(test))
class_names = train.classes

def get_features(dataset):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=100)):
            features = model.encode_image(images.to(device)) #feature shape([100,768])

            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

# Calculate the image features
train_features, train_labels = get_features(train)
val_features, val_labels = get_features(val)
test_features, test_labels = get_features(test)

# Perform logistic regression
classifier = make_pipeline(StandardScaler(), LogisticRegression(random_state=0, C=0.42919342601287785, max_iter=10, verbose=1))
classifier.fit(train_features, train_labels)

# Evaluate using the logistic regression classifier
predictions = classifier.predict(test_features)
print(f1_score(test_labels, predictions, average=None))
print("score=", classifier.score(test_features,test_labels))

accuracy = np.mean((test_labels == predictions).astype(np.float)) * 100.
print(f"Accuracy = {accuracy:.3f}")
print(classification_report(test_labels, predictions, target_names=class_names))

cm = confusion_matrix(test_labels, predictions)
plt.figure(figsize = (10,10))
sn.heatmap(cm, annot=True, cmap="YlGnBu")
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.yticks(np.arange(18)+0.5, class_names, fontsize=12, rotation=0)
plt.xticks([])
plt.savefig("clip_cmai.jpg", bbox_inches='tight')