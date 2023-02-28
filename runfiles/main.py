"""simple run scripts to run the logistic regression on CLIP backbone
"""


import os
import clip
import torch
import torch.utils.data as data

import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from tqdm import tqdm
from sklearn import metrics

from sklearn.metrics import f1_score, classification_report

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-L/14', device)

# Load the dataset
train_data_dir = 'zsf/new_split/percent_split_reduce_subjects/40'
val_data_dir = 'zsf/new_split/val'
train_dataset = datasets.ImageFolder(train_data_dir, transform=preprocess)
val_dataset = datasets.ImageFolder(val_data_dir, transform=preprocess)


class_names = train_dataset.classes

def get_features(dataset):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=100, shuffle=True)):
            features = model.encode_image(images.to(device)) #feature shape([100,768])

            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

# Calculate the image features
train_features, train_labels = get_features(train_dataset)
# val_features, val_labels = get_features(val)
test_features, test_labels = get_features(val_dataset)

# Perform logistic regression & save classifier layer
classifier = LogisticRegression(random_state=0, C=0.42919, max_iter=1000, verbose=1)
classifier.fit(train_features, train_labels)

joblib.dump(classifier, 'model/model_red_new/vitl14_lg40_429_1000.pkl')

# clf = joblib.load('model/lg413935_429_1000.pkl')

# Evaluate using the logistic regression classifier
predictions = classifier.predict(test_features)
f1sc= f1_score(test_labels, predictions, average=None)


# Get predicted probabilities for each class
y_prob = classifier.predict_proba(test_features)

# Get top-3 predicted classes
top3_pred = np.argsort(y_prob, axis=1)[:,:-4:-1]

# Calculate accuracy for top-3 predictions
top3_acc = np.mean(np.any(test_labels.reshape(-1,1) == top3_pred, axis=1)) * 100.

print("avg F1-score=", np.mean(f1sc))

# top3_acc = metrics.top_k_categorical_accuracy(test_labels, predictions, k=3)* 100.
accuracy = np.mean((test_labels == predictions).astype(np.float)) * 100.
print(f"Top-1 acc = {accuracy:.3f}")
print(f"Top-3 acc = {top3_acc:.3f}")

print(classification_report(test_labels, predictions, target_names= class_names))

