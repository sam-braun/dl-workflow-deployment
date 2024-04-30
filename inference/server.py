from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
from io import BytesIO
import base64
import json

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

app = Flask(__name__)

# Load the model
model = Net()

# Load the model state dict
model_state_dict = torch.load('/model/mnist_model.pt')
model.load_state_dict(model_state_dict)
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
        image_file = request.files['image'].read()
        tensor = transform_image(image_bytes=image_file)
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
        response = {'prediction': predicted.item()}
        return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
