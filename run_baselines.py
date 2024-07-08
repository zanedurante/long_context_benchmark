# Blind GPT-4o baseline, based on open-EQA eval

import openai 
from tqdm import tqdm
from gpt4v_video import prompt_gpt4v_images, GLOBAL_TEMPERATURE
from PIL import Image
import base64

with open('keys/openai.key', 'r') as f:
    openai.api_key = f.readline().strip()


def answer_question_eqa(question, model="gpt-4o"):
    response = openai.chat.completions.create(
        model=model,#-turbo",
        temperature=GLOBAL_TEMPERATURE,
        messages=[
                        {
                            "role": "system",
                            "content": 
                            """
                            """
                        },
                        {
                            "role": "user",
                            "content":
                                f"""

You are an intelligent question answering agent. I will ask you questions about stock footage and you must provide an answer.

If the question does not provide enough information to properly answer, provide an appropriate guess.  Maintain strict adherence to the formatting provided in the examples.

Q: What machine is on top of the stove?
A: The microwave
Explanation: stoves are typically found in kitchens and near microwaves.

Q: What is the weather in the beach?
A: The weather is beautiful and sunny
Explanation: videos of beaches are typically sunny and show beautiful weather.

Q: What is the artist using to paint the canvas?
A: The artist is using a paintbrush to paint the canvas
Explanation: painting a canvas typically involves using a paintbrush.

Q: {question}
"""
                        }
                    ]
                )
    return response.choices[0].message.content



def idx2question(idx):
    try:
        with open(f'video_files/{idx}/questions.txt', 'r') as f:
            question = f.read()
    except:
        with open(f'video_files/{idx}/question.txt', 'r') as f:
            question = f.read()
    return question

def idx2video_path(idx):
    return f"video_files/{idx}/combined.mp4"

GPT_4V_PROMPT = """You are an intelligent question answering agent. I will ask you questions about stock footage and you must provide an answer.
You will be shown a set of images that have been collected from a series of stock footage videos.
Given a user query, you must output `text` to answer to the question asked by the user.
User Query: {question}"""


if __name__ == "__main__":
    NUM_FRAMES = 64
    if False: # set to true to run the phi-3-vision baseline
        # TODO: Implement this
        from phi_3 import generate_phi3_response
        idxs = range(100)
        for idx in tqdm(idxs):
            question = idx2question(idx)
            input_text = GPT_4V_PROMPT.replace("{question}", question)
            video_dir = f"video_files/{idx}/{NUM_FRAMES}_frames"
            pred = generate_phi3_response(video_dir, input_text)
            with open(f"video_files/{idx}/{NUM_FRAMES}_frames_phi3_answer.txt", "w") as f:
                f.write(pred + "\n")


    if True: # set to true to run gpt-4v/o with 16 frames baseline
        error_idxs = []
        model = "gpt-4o"
        idxs = range(100)
        for idx in tqdm(idxs):
            question = idx2question(idx)
            path = idx2video_path(idx)
            input_text = GPT_4V_PROMPT.replace("{question}", question)
            video_dir = f"video_files/{idx}/{NUM_FRAMES}_frames"
            try:
                pred = prompt_gpt4v_images(video_dir, input_text, model=model)
            except:
                print("Error with video", idx, "trying again...")
                try:
                    pred = prompt_gpt4v_images(video_dir, input_text, model=model)
                except:
                    print("Error with video", idx, "skipping...")
                    error_idxs.append(idx)
                    continue
            with open(f'video_files/{idx}/{NUM_FRAMES}_frames_{model}_answer.txt', "w") as f:
                f.write(pred + "\n")
        
        print("Error idxs:", error_idxs)


    if False: # set to true to run blind baseline
        for idx in tqdm(range(100)):
            question = idx2question(idx)
            pred = answer_question_eqa(question)
            # Grab from A: to \n
            pred = pred.split('A: ')[1].split('\n')[0]
            with open(f'video_files/{idx}/blind_gpt4o_answer.txt', 'w') as f:
                f.write(pred + '\n')