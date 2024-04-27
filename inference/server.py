from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
from io import BytesIO
import base64
import json

app = Flask(__name__)

# Load the trained model
model = torch.load('/model/mnist_model.pt')
model.eval()

def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = Image.open(BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        image_file = request.files['image'].read()
        tensor = transform_image(image_bytes=image_file)
        outputs = model.forward(tensor)
        _, predicted = torch.max(outputs.data, 1)
        response = {
            'prediction': predicted.item()
        }
        return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
