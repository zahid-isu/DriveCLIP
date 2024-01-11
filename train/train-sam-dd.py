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
data_name = 'sam-dd'
camera_view = ['front_RGB', 'side_RGB']
percentages = [1.0, 0.8, 0.6, 0.4, 0.2] # % of training data to use
"""

# Set training configurations
model_name= 'ViT-L/14'
model_name_path = "".join(model_name.split('/'))
data_name= 'sam-dd'
camera_view ='front_RGB'
percentages = [1.0] # using 100% training data
val_data_dir = f"sam-dd/valid"
train_data_dir = f"sam-dd/train"

# Create lists to store results
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
def is_image_file(filename):
    valid_image_extensions = [".jpg", ".jpeg", ".png"]
    return any(filename.lower().endswith(ext) for ext in valid_image_extensions)

def get_subject_subset(dir, percentage):
    all_subjects = sorted(os.listdir(dir))
    np.random.shuffle(all_subjects)
    subset_size = int(len(all_subjects) * percentage)
    return all_subjects[:subset_size]


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None, is_valid_file=None, included_subjects=None):
        # If no valid file checker is specified, use the is_image_file function
        self.is_valid_file = is_valid_file or is_image_file
        self.included_subjects = included_subjects
        super(CustomImageFolder, self).__init__(root, transform=transform, is_valid_file=self.is_valid_file)
        # Overriding the samples attribute to account for the new directory structure
        self.samples = self.make_dataset_with_camera_views(self.root, is_valid_file=self.is_valid_file)
    
    def make_dataset_with_camera_views(self, dir, is_valid_file=None):
        instances = []
        dir = os.path.expanduser(dir)
        
        # Filter subjects if specified
        all_subjects = sorted(os.listdir(dir))
        subjects = self.included_subjects if self.included_subjects else all_subjects
        
        # Iterate through each subject directory
        for subject_dir in tqdm(sorted(os.listdir(dir)), desc='Subjects'):
            subject_path = os.path.join(dir, subject_dir)
            
            if not os.path.isdir(subject_path):
                continue
            # create class_to_idx inside the method
            class_to_idx = {d: i for i, d in enumerate(sorted(os.listdir(subject_path)))}
            
            # Iterate through each class directory
            for class_name in sorted(class_to_idx.keys()):
                class_index = class_to_idx[class_name]
                class_dir = os.path.join(subject_path, class_name)
                camera_angle_dir = os.path.join(class_dir, "front_RGB")  #Select camera-view ["front_RGB","side_RGB"]
                
                if os.path.isdir(camera_angle_dir):
                    for root, _, fnames in os.walk(camera_angle_dir):
                        for fname in fnames:
                            if is_valid_file(fname):
                                path = os.path.join(root, fname)
                                instances.append((path, class_index))
        return instances
    

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path = self.samples[index][0]
        return img, label, path


# Start training ...
for pct in percentages:
    print(f"Training with {pct*100}% of the data")
    subjects_subset = get_subject_subset(train_data_dir, pct)
    print(subjects_subset)
    print(f"Number of subjects used: {len(subjects_subset)}")

    train_dataset = CustomImageFolder(train_data_dir, transform=preprocess)
    val_dataset = CustomImageFolder(val_data_dir, transform=preprocess)
    class_names = [str(i) for i in range(10)]


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
