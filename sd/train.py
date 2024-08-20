import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import json

from torchvision import transforms
from transformers import CLIPTokenizer
import torch.optim as optim

# Dataset class
class ImageCaptionDataset(Dataset):
    def __init__(self, image_folder, captions_file, tokenizer, transform=None):
        self.image_folder = image_folder
        self.captions = self.load_captions(captions_file)
        self.tokenizer = tokenizer
        self.transform = transform

    def load_captions(self, captions_file):
        with open(captions_file, 'r') as f:
            captions = json.load(f)
        return captions

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image_name = self.captions[idx]['image']
        caption = self.captions[idx]['caption']

        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        tokens = self.tokenizer.batch_encode_plus(
            [caption], padding="max_length", max_length=77
        ).input_ids[0]

        tokens = torch.tensor(tokens, dtype=torch.long)

        return image, tokens

# Define transformations
HEIGHT, WIDTH = 224, 224
transform = transforms.Compose([
    transforms.Resize((HEIGHT, WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

# Initialize the tokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# Instantiate the dataset
dataset = ImageCaptionDataset(
    image_folder="../images/",
    captions_file="../data/captions.json",
    tokenizer=tokenizer,
    transform=transform
)

# Define batch size and DataLoader
batch_size = 8
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Train function
def train(model, dataloader, epochs=10):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        for images, captions in dataloader:
            images = images.to(DEVICE)
            captions = captions.to(DEVICE)

            outputs = model(images, captions)

            loss = loss_fn(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        save_checkpoint(epoch, model, optimizer)

# Save checkpoint function
def save_checkpoint(epoch, model, optimizer, checkpoint_dir='../checkpoints/'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")

# Device selection
DEVICE = "cpu"
ALLOW_CUDA = False
ALLOW_MPS = False

if ALLOW_CUDA and torch.cuda.is_available():
    DEVICE = "cuda"
elif ALLOW_MPS and torch.has_mps and torch.backends.mps.is_available():
    DEVICE = "mps"

print(f"Using device: {DEVICE}")

# Assuming model_loader is your custom model loading module
import model_loader

model_file = "D:\\pytorch-stable-diffusion-main\\pytorch-stable-diffusion-main\\sd\\v1-5-pruned-emaonly.ckpt"
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

# Train the model
train(models, dataloader, epochs=10)



