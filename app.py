from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
from PIL import Image, UnidentifiedImageError
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extensions (only CT scan formats you want to support)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}   # add 'dcm' if you use DICOM

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load trained model
model_path = r'D:\Stroke_Detection-and-Segmentation-by-Using-CNN-ML\all_png_images\EfficiencyNet_stroke_classification_6600.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(weights=None)  # updated to avoid deprecation warning
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class_names = ['non-stroke', 'hemorrhage', 'ischemic']

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/help')
def help_page():
    return render_template('help.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    # Validate file type
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload a CT scan image (PNG, JPG, JPEG).'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # Preprocess image (validate it is a real image)
        image = Image.open(filepath).convert('RGB')
    except UnidentifiedImageError:
        return jsonify({'error': 'Uploaded file is not a valid CT scan image.'}), 400

    input_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)
        label = class_names[pred.item()]

    # For visual result
    if label == 'non-stroke':
        result = {'label': label, 'color': 'green', 'icon': '✅'}
    else:
        result = {'label': label, 'color': 'red', 'icon': '⚠️'}

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
