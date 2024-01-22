"""train scripts for ViT-L/14 on the StateFarm dataset
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

"""
Training configurations:
dataset= StateFarm (https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/data)
clipmodels =['ViT-L/14', 'ViT-L/14@336px','ViT-B/16', 'ViT-B/32', 'RN101']
data_name = 'sf'
percentages = [1.0, 0.8, 0.6, 0.4, 0.2] # % of training data to use
folds_len=7
"""

# Set training configurations and Load the model
model_name= 'ViT-L/14'
model_name_path = "".join(model_name.split('/'))
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-L/14', device)
percentages = [0.4] # using 40% training data
data_name= 'sf'
camera_view ='side_RGB'
acc1=[]
acc3=[]
avgf1=[]


# Load the dataset
train_data_dir = 'zsf/new_split/percent_split_reduce_subjects/40' # 40% of training data
val_data_dir = 'zsf/new_split/val'
train_dataset = datasets.ImageFolder(train_data_dir, transform=preprocess)
val_dataset = datasets.ImageFolder(val_data_dir, transform=preprocess)


class_names = train_dataset.classes

# Start training ...
for pct in percentages:
    print(f"Training with {pct*100}% of the data")

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

    # Get the image features
    train_features, train_labels, train_filepaths = get_features(train_dataset)
    test_features, test_labels, test_filepaths = get_features(val_dataset)

    # Logistic regression
    classifier = LogisticRegression(random_state=0, C=0.42919, max_iter=4000, verbose=0)
    classifier.fit(train_features, train_labels)
    model_path= f"model_ckpt/{data_name}_models/per{pct}_{model_name_path}" # saving checkpoints for each fold

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    joblib.dump(classifier, os.path.join(model_path, f"vitbl14-{data_name}_p{pct}_429_1000.pkl"))

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

    pklfolder_path= f'outputs/pkl/{data_name}_{camera_view}/clip_{model_name_path}/' # create pkl folder path for saving records/predictions
    if not os.path.exists(pklfolder_path):
        os.makedirs(pklfolder_path)

    pklfilePath = pklfolder_path + f"ft-{data_name}_dash-{data_name}-clip_{model_name_path}_p{pct}.pkl"

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

print("Finished training!")

