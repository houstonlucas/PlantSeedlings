# This script takes the data from the plant seedlings kaggle problem and partitions the training data into
# a train and validation set, since the provided test set is unlabeled.
# The script creates two output files containing information for validation and train data.
# Each of these output files contains line separated tuples containing the file path and label for that file
import os
import json

classifications = ["Black-grass",
                   "Charlock",
                   "Cleavers",
                   "Common Chickweed",
                   "Common wheat",
                   "Fat Hen",
                   "Loose Silky-bent",
                   "Maize",
                   "Scentless Mayweed",
                   "Shepherds Purse",
                   "Small-flowered Cranesbill",
                   "Sugar beet"
                   ]

train_list_filename = "train_files"
validation_list_filename = "validation_files"

validation_proportion = 0.1

train_list = {}
validation_list = {}

for classification in classifications:
    dir_name = os.path.join("train", classification)
    for root, dirs, files in os.walk(dir_name):
        n = len(files)
        for i, filename in enumerate(files):
            file_path = os.path.join(root, filename)
            if i < n * validation_proportion:
                train_list[file_path] = classification
            else:
                validation_list[file_path] = classification

with open(train_list_filename, 'w+') as f:
    f.write(json.dumps(train_list, indent=2))

with open(validation_list_filename, 'w+') as f:
    f.write(json.dumps(validation_list, indent=2))

