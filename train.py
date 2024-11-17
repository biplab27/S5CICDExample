import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model.mnist_model import MNISTNet
from datetime import datetime
import os

def train():
    # Force CPU usage
    device = torch.device("cpu")
    
    # Reordered transforms: PIL transforms first, then tensor transforms
    transform = transforms.Compose([
        # PIL Image transforms
        transforms.RandomRotation(15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=(-10, 10)
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        # Convert to tensor
        transforms.ToTensor(),
        # Tensor transforms
        transforms.RandomInvert(p=0.1),
        transforms.RandomErasing(p=0.1),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
    
    # Train for 1 epoch with more detailed progress
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if batch_idx % 100 == 99:
            print(f'Batch {batch_idx+1}/{len(train_loader)}, '
                  f'Loss: {running_loss/100:.4f}')
            scheduler.step(running_loss/100)
            running_loss = 0.0
    
    # Save model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'models/mnist_model_{timestamp}.pth'
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved as {save_path}")

if __name__ == "__main__":
    train() 