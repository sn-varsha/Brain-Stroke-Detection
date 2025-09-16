# import os
# from datasets import load_dataset

# # Load the dataset
# ds = load_dataset('BTX24/tekno21-brain-stroke-dataset-binary')

# # Create output folder
# output_folder = "all_png_images"
# os.makedirs(output_folder, exist_ok=True)

# # Save all images from the train split
# for i, sample in enumerate(ds['train']):
#     img = sample["image"]  # PIL Image
#     label = sample["label"]
#     img.save(f"{output_folder}/{i}_label{label}.png")

# print("All images saved to", output_folder)

import os
import shutil

img_folder = r"D:\Stroke_Detection-and-Segmentation-by-Using-CNN-ML\all_png_images"

label_map = {
    "0": "non-stroke",
    "1": "stroke"
}

# Create subfolders
for class_name in label_map.values():
    os.makedirs(os.path.join(img_folder, class_name), exist_ok=True)

# Move files
for fname in os.listdir(img_folder):
    if fname.endswith(".png") and "label" in fname:
        label = fname.split("label")[1].split(".")[0]
        class_name = label_map.get(label)
        if class_name:
            src = os.path.join(img_folder, fname)
            dst = os.path.join(img_folder, class_name, fname)
            shutil.move(src, dst)

print("Images organized into 'non-stroke' and 'stroke' folders.")