import os
import cv2
import json

# Input and output folders
mask_folder = r'D:\Stroke_Detection-and-Segmentation-by-Using-CNN-ML\ischemic\MASK'  # Folder containing the mask images
json_output_folder = r'D:\Stroke_Detection-and-Segmentation-by-Using-CNN-ML\ischemic\JSON'  # Folder to save the generated JSON files

# Create output folder if it does not exist
os.makedirs(json_output_folder, exist_ok=True)

# Process all mask images
for filename in os.listdir(mask_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        mask_path = os.path.join(mask_folder, filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Thresholding (convert white areas to 1)
        _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # List of shapes for the JSON file
        shapes = []

        for contour in contours:
            contour = contour.squeeze()
            if len(contour.shape) != 2:
                continue  # Skip if the contour does not have enough points
            points = contour.tolist()

            shapes.append({
                "label": "stroke",
                "points": points
            })

        # Define the JSON structure
        json_data = {
            "shapes": shapes,
            "imagePath": filename,
            "imageData": None,
            "imageHeight": mask.shape[0],
            "imageWidth": mask.shape[1]
        }

        # Save the JSON file
        json_filename = os.path.splitext(filename)[0] + ".json"
        json_path = os.path.join(json_output_folder, json_filename)

        with open(json_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)

print("All JSON files of ischemic were successfully created!")
