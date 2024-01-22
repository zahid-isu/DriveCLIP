import os
import shutil
import json
import re
import glob
from pathlib import Path


# Function to read the JSON file 
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Function to group files by subject-ID
def group_files_by_subject_id(files_list):
    subjects_files = {}
    pattern = re.compile(r'sub_(\d+)')
    print("Total files:", len(files_list))

    # Group files by subject-ID
    for file in files_list:
        file_name = os.path.basename(file)
        match = pattern.search(file_name)
        if match:
            subject_id = match.group(1)  # Extracted subject ID
            if subject_id not in subjects_files:
                subjects_files[subject_id] = []
            # Limit the number of files (<= 800) for each subject ID 
            if len(subjects_files[subject_id]) < 800:
                subjects_files[subject_id].append(file)

    return subjects_files


def move_files_to_folds(subject_id_dict, subjects_files, destination, class_map):
    for fold, ids in subject_id_dict.items():
        train_subject_ids = set(map(str, ids['train_sub_id']))
        test_subject_ids = set(map(str, ids['test_sub_id']))

        # Create directories for train and test in each fold
        fold_train_path = os.path.join(destination, fold, "train")
        fold_test_path = os.path.join(destination, fold, "test")
        os.makedirs(fold_train_path, exist_ok=True)
        os.makedirs(fold_test_path, exist_ok=True)

        # Move train files
        for subj_id in train_subject_ids:
            if subj_id in subjects_files:
                for file in subjects_files[subj_id]:
                     # Determine class index from file path
                    class_index = os.path.basename(os.path.dirname(file))
                    class_name = class_map.get(class_index, "unknown")

                    # Create subfolder for the class and copy file
                    class_train_path = os.path.join(fold_train_path, class_name)
                    os.makedirs(class_train_path, exist_ok=True)
                    shutil.copy(file, class_train_path)

        # Move test files
        for subj_id in test_subject_ids:
            if subj_id in subjects_files:
                for file in subjects_files[subj_id]:
                    # Determine class index from file path
                    class_index = os.path.basename(os.path.dirname(file))
                    class_name = class_map.get(class_index, "unknown")

                    # Create subfolder for the class and copy file
                    class_test_path = os.path.join(fold_test_path, class_name)
                    os.makedirs(class_test_path, exist_ok=True)
                    shutil.copy(file, class_test_path)

def count_files_per_subject(subjects_files):
    file_counts = {}
    for subject_id, files in subjects_files.items():
        file_counts[subject_id] = len(files)
    return file_counts


class_map = {
    'drinking': '0',
    'hair_and_makeup':'1',
    'phonecall_right': '2',
    'radio':'3',
    'reach_backseat': '4',
    'reach_side': '5',
    'safe_drive':'6',
    'talking_to_passenger':'7',
    'texting_right':'8',
    'yawning':'9'
}

# Current paths
json_file_path = 'sub_split_dmd_29.json'
subject_id_dict = read_json_file(json_file_path)
destination = 'data/new' 

data_path = 'data/frame/dmd_rgb_gA/driver_actions/'  # data path gA / gB
class_path= glob.glob(data_path +'/*')
print(class_path)

for i in range(len(class_map)):
    print("Copying class...", class_path[i].split('/')[-1])
    files_list = glob.glob(class_path[i] + '/*.jpg')
    # Group files by subject-ID
    subjects_files = group_files_by_subject_id(files_list)
    file_counts = count_files_per_subject(subjects_files)
    # Print the counts
    for subject_id, count in file_counts.items():
        print(f"Subject ID {subject_id} has {count} files.")
    move_files_to_folds(subject_id_dict, subjects_files, destination, class_map)