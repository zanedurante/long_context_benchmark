import openai
from openai import OpenAI
import base64
import os

with open('keys/openai.key', 'r') as f:
    openai.api_key = f.readline().strip()


client = OpenAI(
    api_key=openai.api_key
)

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


for i in [1, 2, 3, 4, 5]:
    base64_image = encode_image(f"ex{i}.png")

    response = client.chat.completions.create(
    temperature=0.0,
    model="gpt-4-turbo",
    messages=[
        {
        "role": "user",
        "content": [
            {"type": "text", "text": "You are an intensivist physician working in an ICU.  You are interested in assessing a patient’s pulmonary clinical status and how it is changing over time in order to decide what is the best next action (therapeutic or diagnostic) to take.  The only observational modality available to you is camera vision.   Focus on the interactions of the clinician with the patient and the patient’s activities to assess the patient’s pulmonary clinical status.  If you are uncertain about the pulmonary clinical status based on the single frame, pose a visual-modality question you would like to ask in order to increase the confidence in your assessment.  This may involve asking for a frame that precedes or follows this image in time.  Be specific about what activity or change in activity you are looking for in a previous or following frame."},
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            },
        ],
        }
    ],
    max_tokens=1000,
    )

    os.makedirs("test_output", exist_ok=True)
    with open(f"test_outputs/ex{i}_gpt4v.txt", "w") as f:
        f.write(response.choices[0].message.content)
