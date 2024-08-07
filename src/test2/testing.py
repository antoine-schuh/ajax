import torch
import torchvision.transforms as transforms
from PIL import Image
from simple_cnn import SimpleCNN
from image_transformer import transform


def main():
    # Load the trained model
    model = SimpleCNN()
    model.load_state_dict(torch.load('animals.pth'))
    model.eval()

    # Load and preprocess the image
    image_path = '/users/aschuh/Downloads/1606309216352.jpg'
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Run inference
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    # Define class labels
    classes = ('butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel')

    # Print the predicted class
    print(f'Predicted: {classes[predicted.item()]}')


if __name__ == '__main__':
    main()
