import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from simple_cnn import SimpleCNN
from image_transformer import transform


def main():
    trainset = ImageFolder(root='/users/aschuh/desktop/animals/train', transform=transform)
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    # Print the class names in the order they are indexed by the dataset
    print('Class names in order:', trainset.classes)

    # Create a model instance
    net = SimpleCNN()

    # Define a loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Train the model
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    # Save the trained model
    torch.save(net.state_dict(), 'animals.pth')


if __name__ == '__main__':
    main()
