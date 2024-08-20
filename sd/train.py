import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import json
import torch
import numpy as np
from tqdm import tqdm
import json
from ddpm import DDPMSampler
from torchvision import transforms
from transformers import CLIPTokenizer
import torch.optim as optim
from ddpm import DDPMSampler
n_inference_steps = 2  # Define the number of inference steps
do_cfg = True  # Whether to use classifier-free guidance
cfg_scale = 7.5  # Guidance scale

def get_time_embedding(timestep):
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    # Shape: (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
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
HEIGHT, WIDTH = 512, 512
transform = transforms.Compose([
    transforms.Resize((HEIGHT, WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

# Initialize the tokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# Instantiate the dataset
dataset = ImageCaptionDataset(
    image_folder="/content/pytorch-stable-diffusion/sd/images/",
    captions_file="/content/pytorch-stable-diffusion/sd/data/captions.json",
    tokenizer=tokenizer,
    transform=transform
)

# Define batch size and DataLoader
#batch_size = 4
#dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
batch_size = 1  # Try reducing the batch size to 4 or lower
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
# Train function
import torch.optim as optim

# Updated transformations for the dataset
HEIGHT, WIDTH = 512, 512  # Updated to match encoder's expected input size
transform = transforms.Compose([
    transforms.Resize((HEIGHT, WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

# Updated latents_shape calculation inside the train function
def train(models, dataloader, epochs=10, device="cuda", seed=42):
    # Ensure all models are in training mode
    for model in models.values():
        model.train()
    
    optimizer = optim.Adam(
        [
            {"params": models["diffusion"].parameters()},
            {"params": models["decoder"].parameters()},
        ],
        lr=1e-4
    )
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        epoch_loss = 0
        for images, captions in dataloader:
            # Move images and captions to the appropriate device
            images = images.to(device)
            captions = captions.to(device)
            
            # Encode the caption using the CLIP model to obtain the context
            clip = models["clip"]
            cond_context = clip(captions)
            print(cond_context)
            print(cond_context.shape)

            # Encode the image to obtain the latent representation
            encoder = models["encoder"]
            latents_shape = (images.size(0), 4, HEIGHT // 8, WIDTH // 8)  # Updated shape
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
            encoder_noise = torch.randn(latents_shape, device=device)
            latents = encoder(images, encoder_noise)
            print(latents.shape)

            # Initialize the DDPMSampler
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
            timesteps = sampler.timesteps

            # Iterate over timesteps
            for i, timestep in enumerate(timesteps):
                time_embedding = get_time_embedding(timestep).to(device)

                # Model input
                model_input = latents
                if do_cfg:
                    model_input = model_input.repeat(2, 1, 1, 1)

                # Ensure diffusion model is defined
                diffusion = models["diffusion"]

                # model_output is the predicted noise
                model_output = diffusion(model_input, cond_context, time_embedding)

                if do_cfg:
                    output_cond, output_uncond = model_output.chunk(2)
                    model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

                # Perform denoising using the diffusion model
                latents = sampler.step(timestep, latents, model_output)

            # Decode the latent representation to get the generated image
            decoder = models["decoder"]
            generated_images = decoder(latents)

            # Compute the loss between the generated images and the original images
            loss = loss_fn(generated_images, images)
            epoch_loss += loss.item()

            # Backpropagation and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}")
        save_checkpoint(epoch, models, optimizer)

def save_checkpoint(epoch, models, optimizer, checkpoint_dir='../checkpoints/'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'diffusion_state_dict': models['diffusion'].state_dict(),
        'decoder_state_dict': models['decoder'].state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")

# Assuming you have already initialized the models and dataloader:
# models = {
#     "clip": clip_model,
#     "encoder": encoder_model,
#     "diffusion": diffusion_model,
#     "decoder": decoder_model
# }
# dataloader = your_dataloader




# Device selection
DEVICE = "cuda"
ALLOW_CUDA = False
ALLOW_MPS = False

if ALLOW_CUDA and torch.cuda.is_available():
    DEVICE = "cuda"
elif ALLOW_MPS and torch.has_mps and torch.backends.mps.is_available():
    DEVICE = "mps"

print(f"Using device: {DEVICE}")

# Assuming model_loader is your custom model loading module
import model_loader

model_file = "/content/drive/MyDrive/Fast-Dreambooth/Sessions/surya/surya.ckpt"
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

# Train the model
train(models, dataloader, epochs=10,seed=42)



