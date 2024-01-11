import csv
import clip
import joblib
import numpy as np
import os
import pickle
from sklearn import metrics
from sklearn.metrics import f1_score, classification_report
from sklearn.linear_model import LogisticRegression
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import make_dataset
from tqdm import tqdm

"""
Training configurations:
clipmodels =['ViT-L/14', 'ViT-L/14@336px','ViT-B/16', 'ViT-B/32', 'RN101']
data_name = 'dmd29'
camera_view = ['front_RGB', 'side_RGB']
percentages = [1.0, 0.8, 0.6, 0.4, 0.2] # % of training data to use
folds_len=7
"""


# Select model and data-fps:
model_name= 'ViT-L/14'
model_name_path = "".join(model_name.split('/'))
data_name= 'dmd29'
folds_len=7

acc1=[]
acc3=[]
avgf1=[]
overall_acc1=[]
overall_acc3=[]
overall_f1=[]
results = []

# Load the model to cuda
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(model_name, device) # load CLIP model backbone (ViT encoders)


# Load the dataset
for i in range (len(folds_len)): # 7-fold cross-validation
    val_data_dir = f"dmd/DMD-Driver/data/new/fold{i}/test"
    train_data_dir = f"dmd/DMD-Driver/data/new/fold{i}/train"


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
            for  batch_idx, (images, labels, paths) in enumerate(DataLoader(dataset, batch_size=512, shuffle=True)):
                features = model.encode_image(images.to(device)) #feature shape([100,768])
                filepaths += (paths)

                all_features.append(features)
                all_labels.append(labels)

        return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy(),  filepaths

    # Calculate the image features
    train_features, train_labels, train_filepaths = get_features(train_dataset)
    test_features, test_labels, test_filepaths = get_features(val_dataset)

    # Perform logistic regression
    classifier = LogisticRegression(random_state=0, C=0.42919, max_iter=4000, verbose=0)
    classifier.fit(train_features, train_labels)
    model_path= f"model_ckpt/{data_name}_models/{model_name_path}" # saving checkpoints for each fold

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    joblib.dump(classifier, os.path.join(model_path, f"{data_name}_vitbl14-fold{i}_429_1000.pkl"))

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

    pklfolder_path= f'outputs/pkl/{data_name}/clip_{model_name_path}/' # create pkl folder path for saving records/predictions
    if not os.path.exists(pklfolder_path):
        os.makedirs(pklfolder_path)

    pklfilePath = pklfolder_path + f"ft-{data_name}_dash-fold{i}-clip_{model_name_path}.pkl"

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


    fold_results = {
        'fold': f"fold{i}",
        'top1_acc': accuracy,
        'top3_acc': top3_acc,
        'avg_f1_score': avf1
    }
    results.append(fold_results)

    print(f"model finished for fold{i}")


# Writing results to a CSV file
csv_file = 'dmd29_training_results.csv'
with open(csv_file, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['fold', 'top1_acc', 'top3_acc', 'avg_f1_score'])
    writer.writeheader()
    writer.writerows(results)

print("All folds processed. Results saved to", csv_file)