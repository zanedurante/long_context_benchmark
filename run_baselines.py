# Blind GPT-4o baseline, based on open-EQA eval

import openai 
from tqdm import tqdm
from gpt4v_video import prompt_gpt4v_images, GLOBAL_TEMPERATURE
from PIL import Image
import base64
import google.generativeai as genai
import os
from time import sleep


with open('keys/openai.key', 'r') as f:
    openai.api_key = f.readline().strip()

with open('keys/gemini.key', 'r') as f:
    genai.configure(api_key=f.readline().strip())


gemini_safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
    },
]

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


GPT_4V_PROMPT = """You are an intelligent question answering agent. I will ask you questions about stock footage and you must provide an answer.
You will be shown a set of images that have been collected from a series of stock footage videos.
Given a user query, you must output `text` to answer to the question asked by the user.
User Query: {question}"""


if __name__ == "__main__":
    NUM_FRAMES = 16 # Integer 2-64 or "all" which uses model-specific sampling (e.g. gemini)
    special_idxs = [] # If [] then all videos are evaluated, otherwise only the special_idxs are evaluated
    if len(special_idxs) == 0:
        idxs = range(100)
    else:
        idxs = special_idxs
    if False: # set to true to run the phi-3-vision baseline
        # TODO: Implement this
        from phi_3 import generate_phi3_response
        for idx in tqdm(idxs):
            question = idx2question(idx)
            input_text = GPT_4V_PROMPT.replace("{question}", question)
            video_dir = f"video_files/{idx}/{NUM_FRAMES}_frames"
            pred = generate_phi3_response(video_dir, input_text)
            with open(f"video_files/{idx}/{NUM_FRAMES}_frames_phi3_answer.txt", "w") as f:
                f.write(pred + "\n")

    if True: # Set to true to run gemini 1.5 pro/flash, need to activate google environment
        error_idxs = []
        model = "gemini-1.5-flash" # Alternatively, gemini-1.5-flash, gemini-1.5-pro
        gen_model = genai.GenerativeModel(model)

        for idx in tqdm(idxs):
            # check if output already exists
            if os.path.exists(f'video_files/{idx}/{NUM_FRAMES}_frames_{model}_answer.txt'):
                continue
            question = idx2question(idx)
            input_text = GPT_4V_PROMPT.replace("{question}", question)

            if type(NUM_FRAMES) == int:
                video_dir = f"video_files/{idx}/{NUM_FRAMES}_frames"
                frames = os.listdir(video_dir)
                frames = [f for f in frames if f.endswith('.mp4')] # change to .jpg for frame-level analysis
                frames = sorted([f"{video_dir}/{frame}" for frame in frames])

            elif NUM_FRAMES == "all":
                video_dir = f"video_files/{idx}"
                frames = [file for file in os.listdir(video_dir) if file.endswith('.mp4')]
                frames = sorted([f"{video_dir}/{frame}" for frame in frames])


            # TODO: Try with get_file? https://ai.google.dev/api/python/google/generativeai
            uploaded_files =[]
            for frame in frames:
                uploaded_file = genai.upload_file(path=frame, display_name=frame)
                uploaded_files.append(uploaded_file)
            
            entire_input = [*uploaded_files, input_text]
            if type(NUM_FRAMES) == str:
                # Wait for a while to allow for the video to be uploaded to google genai
                sleep(20)
            try:
                pred = gen_model.generate_content(entire_input, safety_settings=gemini_safety_settings).text
            except Exception as e:
                print(e)
                print("Error with video", idx, "trying again...")
                
                try:
                    pred = gen_model.generate_content(entire_input).text
                except:
                    print("Error with video", idx, "skipping...")
                    error_idxs.append(idx)
                    pred = "Sorry, I cannot answer this question."
            with open(f'video_files/{idx}/{NUM_FRAMES}_frames_{model}_answer.txt', "w") as f:
                f.write(pred + "\n")
            
        print("Error idxs:", error_idxs)


    if False: # set to true to run gpt-4v/o with NUM_FRAMES frames baseline
        error_idxs = []
        model = "gpt-4o"
        for idx in tqdm(idxs):
            question = idx2question(idx)
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
        for idx in tqdm(idxs):
            question = idx2question(idx)
            pred = answer_question_eqa(question)
            # Grab from A: to \n
            pred = pred.split('A: ')[1].split('\n')[0]
            with open(f'video_files/{idx}/blind_gpt4o_answer.txt', 'w') as f:
                f.write(pred + '\n')