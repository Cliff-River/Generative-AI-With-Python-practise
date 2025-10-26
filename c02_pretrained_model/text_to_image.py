import torch
from diffusers import AmusedPipeline

pipe = AmusedPipeline.from_pretrained("amused/amused-256", variant="fp16", torch_dtype=torch.float16)
pipe.vqvae.to(torch.float32)
pipe = pipe.to("cuda")

prompt = "a little cat"
image = pipe(prompt, generator=torch.Generator("cuda").manual_seed(8)).images[0]
image.save("cat.png")