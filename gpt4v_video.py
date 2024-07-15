import cv2
import base64
import openai
import copy
import os
import random
from tqdm import tqdm

GLOBAL_TEMPERATURE = 0.2

with open('keys/openai.key', 'r') as f:
    openai.api_key = f.readline().strip()

def encode_frame(frame):
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")

def encode_frame_from_image_path(image_path):
    frame = cv2.imread(image_path)
    return encode_frame(frame)

def get_video_length(video):
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    return length

def get_video_idxs(video, num_idxs):
    # Linearly spaced indices (uniformly spaced)
    length = get_video_length(video)
    distance = length // num_idxs
    offset = distance // 2
    return [int(offset + distance * i) for i in range(num_idxs)]



def get_video_frames(video_path, num_frames=16):
    video = cv2.VideoCapture(video_path)
    frames = []
    idxs = get_video_idxs(video, num_frames)
    prev_frame = None
    for idx in idxs:
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = video.read()
        if not success:
            print("Error reading frame", idx) # replace current frame with frame + 1
            print("Using previous frame instead (if availabe)")
            if prev_frame is not None:
                frames.append(prev_frame)
        else:
            encoded_frame = encode_frame(frame)
            frames.append(encoded_frame)
            prev_frame = encoded_frame

    return frames


def prompt_gpt4v_images(frame_dir, text_prompt, model="gpt-4o"):
    frames = os.listdir(frame_dir)
    frames = sorted([f"{frame_dir}/{frame}" for frame in frames if frame.endswith('.jpg')])

    encoded_frames = [encode_frame_from_image_path(frame) for frame in frames]

    response = openai.chat.completions.create(
        model=model,#-turbo",
        temperature=GLOBAL_TEMPERATURE,
        messages=[
                        {
                            "role": "system",
                            "content": "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture. Knowledge cutoff: 2023-10 Current date: 2024-05-04",
                            "role": "user",
                            "content": [
                                text_prompt,
                                *map(lambda x: {"image": x, "resize": 768}, encoded_frames),
                            ],
                        }
                    ]
                )
    return response.choices[0].message.content

def prompt_gpt4v(video_path, text_prompt, model="gpt-4o", num_frames=16):
    response = openai.chat.completions.create(
        model=model,#-turbo",
        temperature=GLOBAL_TEMPERATURE,
        messages=[
                        {
                            "role": "system",
                            "content": "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture. Knowledge cutoff: 2023-10 Current date: 2024-05-04",
                            "role": "user",
                            "content": [
                                text_prompt,
                                *map(lambda x: {"image": x, "resize": 768}, get_video_frames(video_path, num_frames=num_frames)),
                            ],
                        }
                    ]
                )
    return response.choices[0].message.content