import os
import clip
import torch
import torch.utils.data as data

import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder

from tqdm import tqdm
from sklearn import metrics

from sklearn.metrics import f1_score, classification_report
import pickle
"""
Training script of  frame-based CLIP model.
clipmodels =['ViT-L/14', 'ViT-B/16', 'ViT-B/32', 'RN101']
data-fps= [1,2,5,10,20]
"""

# Select model and data-fps:
model_name= 'RN101'
model_name_path = "".join(model_name.split('/'))
data_fps= 'syn2fps'


acc1=[]
acc3=[]
avgf1=[]
overall_acc1=[]
overall_acc3=[]
overall_f1=[]

# Run the frame-based CLIP model and save the results
# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(model_name, device) # load CLIP model backbone (ViT encoders)

# Load the dataset

val_data_dir = f"data/{data_fps}/val"
train_data_dir = f"data/{data_fps}/train"


class CustomImageFolder(ImageFolder):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path = self.imgs[index][0]
        return img, label, path

train_dataset = CustomImageFolder(train_data_dir, transform=preprocess)
val_dataset = CustomImageFolder(val_data_dir, transform=preprocess)



class_names = train_dataset.classes

def get_features(dataset):
    all_features = []
    all_labels = []
    filepaths = []

    with torch.no_grad():
        for  batch_idx, (images, labels, paths) in enumerate(DataLoader(dataset, batch_size=100, shuffle=True)):
            features = model.encode_image(images.to(device)) #feature shape([100,768])
            filepaths += (paths)

            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy(),  filepaths

# Calculate the image features
train_features, train_labels, train_filepaths = get_features(train_dataset)
test_features, test_labels, test_filepaths = get_features(val_dataset)

# Perform logistic regression
classifier = LogisticRegression(random_state=0, C=0.42919, max_iter=1000, verbose=0)
classifier.fit(train_features, train_labels)
model_path= f"model/{data_fps}_models/{model_name_path}" # saving checkpoints for each fold

if not os.path.exists(model_path):
    os.makedirs(model_path)

joblib.dump(classifier, f"model/{data_fps}_models/{model_name_path}/vitbl14-per1_429_1000.pkl")

# Evaluate using the logistic regression classifier
predictions = classifier.predict(test_features)
f1sc= f1_score(test_labels, predictions, average=None)


# Get predicted probabilities for each class
probs = classifier.predict_proba(test_features)

pkl_records = {
    'probs':probs,
    'labels':test_labels,
    'imgPath': test_filepaths
}

pklfolder_path= f'outputs/pkl/{data_fps}/clip_{model_name_path}/' # create pkl folder path for saving records/predictions
if not os.path.exists(pklfolder_path):
    os.makedirs(pklfolder_path)

pklfilePath = pklfolder_path + f"ft-{data_fps}_dash-1.0-clip_{model_name_path}-0.001.pkl"

with open(pklfilePath, 'wb') as f:
    pickle.dump(pkl_records, f)

# Get top-3 predicted classes
top3_pred = np.argsort(probs, axis=1)[:,:-4:-1]

# Calculate accuracy for top-3 predictions
top3_acc = np.mean(np.any(test_labels.reshape(-1,1) == top3_pred, axis=1)) * 100.
acc3.append(top3_acc)
avf1= np.mean(f1sc)
avgf1.append(avf1)

accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
acc1.append(accuracy)
print(f"Top-1 acc = {accuracy:.3f}")
print(f"Top-3 acc = {top3_acc:.3f}")
print("avg F1-score=", avf1)
print(classification_report(test_labels, predictions, target_names= class_names))


print("model finished")
