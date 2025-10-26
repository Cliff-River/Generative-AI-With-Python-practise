#%%
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import base64
from groq import Groq

load_dotenv(find_dotenv())

# %%
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

current_dir = Path(__file__).parent.absolute()
base64_image = encode_image(current_dir / "sample_image.png")
#%%

client = Groq()

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        }
    ],
    model="meta-llama/llama-4-scout-17b-16e-instruct",
)

print(chat_completion.choices[0].message.content)

#%%

