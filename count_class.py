import os  # for working with folders/files

train_dir = "Train"  # path to Train folder, update if needed

class_counts = {}  # dictionary to store count per class

for class_folder in sorted(os.listdir(train_dir), key=lambda x: int(x)):  # loop through class folders in order
    class_path = os.path.join(train_dir, class_folder)  # full path to this class folder
    if os.path.isdir(class_path):  # make sure it's actually a folder
        num_images = len(os.listdir(class_path))  # count files inside
        class_counts[class_folder] = num_images  # save count for this class

# Print counts
for class_id, count in class_counts.items():  # go through each class and its count
    print(f"Class {class_id}: {count} images")  # show count per class

print(f"\nTotal images: {sum(class_counts.values())}")  # sum of all images
print(f"Max: {max(class_counts.values())}, Min: {min(class_counts.values())}")  # biggest and smallest class