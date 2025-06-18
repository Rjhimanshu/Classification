from flask import Flask, request, render_template, jsonify
import os
import torch
from PIL import Image
import io
from torchvision import transforms
from transformers import ViTForImageClassification, ViTFeatureExtractor

app = Flask(__name__)

# Load model once when app starts
def load_model():
    model_path = 'multiclass_models/best_model.pt'
    info_path = 'multiclass_models/model_info.pt'
    
    model_info = torch.load(info_path)
    class_to_idx = model_info['class_to_idx']
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=len(class_to_idx)
    )
    
    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    return model, feature_extractor, idx_to_class, device

model, feature_extractor, idx_to_class, device = load_model()

def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor).logits
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
    
    return idx_to_class[predicted_class.item()], confidence.item()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        
        if file:
            img_bytes = file.read()
            prediction, confidence = predict_image(img_bytes)
    
    return render_template('index.html', 
                          prediction=prediction, 
                          confidence=confidence)

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.mkdir('templates')
    app.run(debug=True)