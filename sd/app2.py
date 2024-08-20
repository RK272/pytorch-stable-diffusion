from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
from transformers import CLIPTokenizer
import torch
import model_loader
import pipeline
from pydantic import BaseModel
import io
import base64
import numpy as np

from pyngrok import conf, ngrok

app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API Server"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Or ["*"] to allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the device
DEVICE = "cuda"
ALLOW_CUDA = False
ALLOW_MPS = False

from fastapi.staticfiles import StaticFiles
import os

# Create a static directory if it doesn't exist
if not os.path.exists("static"):
    os.makedirs("static")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")


if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif torch.backends.mps.is_built() and ALLOW_MPS:
    DEVICE = "cpu"
print(f"Using device: {DEVICE}")

# Load the tokenizer and models
tokenizer = CLIPTokenizer("/content/tokenizer_vocab.json", merges_file="/content/tokenizer_merges.txt")
model_file = "/content/surya.ckpt"
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

class PromptData(BaseModel):
    prompt: str
    uncond_prompt: str = ""  # Optional negative prompt
    strength: float = 0.9  # Default strength
    sampler: str = "ddpm"
    num_inference_steps: int = 1
    seed: int = 42


@app.post("/generate_image")
async def generate_image(prompt_data: PromptData):
    print(prompt_data.prompt)
    try:
        output_image = pipeline.generate(
            prompt=prompt_data.prompt,
            uncond_prompt=prompt_data.uncond_prompt,
            input_image=None,  # Handle image uploads if needed
            strength=prompt_data.strength,
            do_cfg=True,
            cfg_scale=8,  # Adjust as needed
            sampler_name=prompt_data.sampler,
            n_inference_steps=prompt_data.num_inference_steps,
            seed=prompt_data.seed,
            models=models,
            device=DEVICE,
            idle_device="cpu",
            tokenizer=tokenizer,
        )
        output_image_path = os.path.join("static", "output_image.png")
        Image.fromarray(output_image).save(output_image_path)

        np_image = np.array(output_image)

        # Convert NumPy array to bytes
        image_bytes = io.BytesIO()
        Image.fromarray(np_image).save(image_bytes, format='PNG')
        image_bytes.seek(0)

        # Encode bytes to base64
        encoded_image = base64.b64encode(image_bytes.read()).decode('utf-8')

        return {"status": "success", "image_data": encoded_image}

        return {"status": "success", "image_path": f"/static/output_image.png"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

       

# Set up ngrok to create a tunnel to the localhost server
#public_url = ngrok.connect(port=8000)
ngrok_config = conf.PyngrokConfig(region="us")  # Adjust the region as needed
public_url = ngrok.connect(8000, pyngrok_config=ngrok_config)
print(f"Your public URL is: {public_url}")

# Start the server
uvicorn.run(app, host="0.0.0.0", port=8000)
