import pandas as pd  # to read the CSV files
import numpy as np  # for array operations
from PIL import Image  # to load and resize images
import os  # for file paths

DATASET_DIR = "dataset"  # base folder where Train/Test/Meta live

train_df = pd.read_csv(os.path.join(DATASET_DIR, "Train.csv"))  # load training metadata (paths + labels)

IMG_SIZE = 32  # resize all images to 32x32 (standard for GTSRB)

images = []  # will hold image arrays
labels = []  # will hold class labels

for idx, row in train_df.iterrows():  # loop through every row in Train.csv
    img_path = os.path.join(DATASET_DIR, row["Path"])  # prepend dataset folder to the relative path
    img = Image.open(img_path).convert("RGB")  # open image and force RGB
    img = img.resize((IMG_SIZE, IMG_SIZE))  # resize to fixed size
    images.append(np.array(img))  # convert to array and store
    labels.append(row["ClassId"])  # store corresponding class label

X = np.array(images, dtype="float32") / 255.0  # normalize pixel values to 0-1
y = np.array(labels)  # convert labels list to array

print("X shape:", X.shape)  # sanity check: should be (num_images, 32, 32, 3)
print("y shape:", y.shape)  # sanity check: should be (num_images,)