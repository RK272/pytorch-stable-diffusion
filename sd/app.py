from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
from transformers import CLIPTokenizer
import torch
import model_loader
import pipeline
from pydantic import BaseModel

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
DEVICE = "cpu"
ALLOW_CUDA = False
ALLOW_MPS = False

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "cpu"
print(f"Using device: {DEVICE}")

# Load the tokenizer and models
tokenizer = CLIPTokenizer("tokenizer_vocab.json", merges_file="tokenizer_merges.txt")
model_file = "surya.ckpt"
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

        output_image_path = "output_image.png"
        Image.fromarray(output_image).save(output_image_path)

        return {"status": "success", "image_path": output_image_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
