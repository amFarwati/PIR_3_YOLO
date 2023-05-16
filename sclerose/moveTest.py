#
# This script moves 20% of the data from train directory to test directory
# Not necessary if already done !! 
#
import os
import random
import shutil

train_dir = 'data/MICCAI/train'
test_dir = 'data/MICCAI/test'

train_flair_dir = os.path.join(train_dir, 'flair')
train_seg_dir = os.path.join(train_dir, 'seg')

test_flair_dir = os.path.join(test_dir, 'flair')
test_seg_dir = os.path.join(test_dir, 'seg')

# Create test directories if they don't exist
os.makedirs(test_flair_dir, exist_ok=True)
os.makedirs(test_seg_dir, exist_ok=True)

# Get the list of flair and seg files
flair_files = os.listdir(train_flair_dir)
seg_files = os.listdir(train_seg_dir)

# Randomly select 20% of the data for the test set
test_size = int(0.2 * len(flair_files))

# Shuffle the list of files
file_indices = list(range(len(flair_files)))
random.shuffle(file_indices)

# Select files for the test set
test_files = file_indices[:test_size]

for index in test_files:
    # Get the corresponding filenames
    flair_file = flair_files[index]
    seg_file = seg_files[index]

    # Move the files to the test directories
    shutil.move(
        os.path.join(train_flair_dir, flair_file),
        os.path.join(test_flair_dir, flair_file)
    )
    shutil.move(
        os.path.join(train_seg_dir, seg_file),
        os.path.join(test_seg_dir, seg_file)
    )