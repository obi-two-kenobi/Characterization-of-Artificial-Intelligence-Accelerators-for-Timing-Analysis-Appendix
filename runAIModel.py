import sys
sys.path.append("brevitas/src/brevitas_examples")  # Add the path to the Brevitas examples to the system path
from bnn_pynq.models.CNV import*  # Import the CNV model from the Brevitas examples

# Configuration
import configparser
config = configparser.ConfigParser()
config['QUANT'] = {'WEIGHT_BIT_WIDTH': '1',  # Set weight bit width to 1
                   'ACT_BIT_WIDTH': '1',     # Set activation bit width to 1
                   'IN_BIT_WIDTH': '8'}      # Set input bit width to 8
config['MODEL'] = {'NUM_CLASSES':'2',        # Set number of output classes to 2
                   'IN_CHANNELS':'3'}        # Set number of input channels to 3

# Model
model = cnv(config)  # Initialize the CNV model with the configuration

import torch
from torchvision import transforms
from PIL import Image
import time

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = cnv(config).to(device)  # Move the model to the selected device

# Load the trained model weights
model.load_state_dict(torch.load('cnv_4bit_2_output_class.pth', map_location=device))

# Image transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize the image to 32x32 pixels
    transforms.ToTensor(),        # Convert the image to a tensor
])

# Load image
image_path = 'test.png'
image = Image.open(image_path).convert('RGB')  # Open the image and convert it to RGB
image = transform(image)  # Apply the transformations to the image
image = image.unsqueeze(0).to(device)  # Add batch dimension and move to the selected device

# Model inference
start_time = time.time()  # Record the start time
model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # Disable gradient calculation
    output = model(image)  # Perform inference
    _, predicted = torch.max(output, 1)  # Get the predicted class
end_time = time.time()  # Record the end time

# Print the inference time and predicted class
print(f'Inference time: {end_time - start_time} seconds')
print(f'Predicted class: {predicted.item()}')
