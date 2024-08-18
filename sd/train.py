import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import mlflow
import numpy as np
from tqdm import tqdm

# Custom imports for encoder, decoder, and diffusion model
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import Diffusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations for images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load dataset (using an example dataset here, replace with your own)
dataset = ImageFolder(root='D:\\stabgit\\pytorch-stable-diffusion\\sd\\data', transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)


class StableDiffusionModel(nn.Module):
    def __init__(self):
        super(StableDiffusionModel, self).__init__()
        self.encoder = VAE_Encoder()
        self.decoder = VAE_Decoder()
        self.diffusion = Diffusion()

    def forward(self, x):
        latent = self.encoder(x)
        diffused_latent = self.diffusion(latent)
        reconstructed = self.decoder(diffused_latent)
        return reconstructed

model = StableDiffusionModel().to(device)


# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


def train_model(model, data_loader, criterion, optimizer, epochs=10):
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, _) in enumerate(tqdm(data_loader)):
            images = images.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, images)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Log the epoch loss
        epoch_loss = running_loss / len(data_loader)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}')
        
        mlflow.log_metric('epoch_loss', epoch_loss, step=epoch + 1)

    print('Training complete')
mlflow.set_experiment('stable_diffusion_training')

with mlflow.start_run():
    # Log hyperparameters
    mlflow.log_param('learning_rate', 1e-4)
    mlflow.log_param('batch_size', 32)
    mlflow.log_param('epochs', 10)
    
    # Training the model
    train_model(model, data_loader, criterion, optimizer, epochs=10)
    
    # Log the model
    mlflow.pytorch.log_model(model, "stable_diffusion_model")
# Save the final model
torch.save(model.state_dict(), 'stable_diffusion_model.pth')
