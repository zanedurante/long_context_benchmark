from PIL import Image
import requests
from transformers import AutoModelForCausalLM, AutoProcessor
from gpt4v_video import GLOBAL_TEMPERATURE
import os

# Example prompt with 1 image: "<|user|>\n<|image_1|>\n{prompt}<|end|>\n<|assistant|>\n"
def generate_phi3_prompt(num_frames, text_prompt):
    image_str = ""
    for i in range(1, num_frames+1):
        image_str = image_str + "<|image_" + str(i) + "|>"
    return "<|user|>\n" + image_str + "\n" + text_prompt + "<|end|>\n<|assistant|>\n"

model_id = "microsoft/Phi-3-vision-128k-instruct" 

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", trust_remote_code=True, torch_dtype="auto", _attn_implementation='flash_attention_2') # use _attn_implementation='eager' to disable flash attention

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True) 

def generate_phi3_response(image_dir, text_input):
    # Loads all images from image_dir into a list of PIL images and pass to phi-3
    images = []
    image_files = sorted(os.listdir(image_dir))
    image_files = [os.path.join(image_dir, image_file) for image_file in image_files]
    for image_file in image_files:
        image = Image.open(image_file)
        images.append(image)

    prompt = generate_phi3_prompt(len(images), text_input)
    
    inputs = processor(prompt, images, return_tensors="pt").to("cuda:0")


    generation_args = { 
        "max_new_tokens": 500, 
        "temperature": GLOBAL_TEMPERATURE, 
        "do_sample": True, 
    } 

    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 

    # remove input tokens 
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 
    return response


if __name__ == "__main__":
    sample_dir = "video_files/0/16_frames"
    text_input = "What food is being cut in the video?"

    response = generate_phi3_response(sample_dir, text_input)
    print("Phi-3 response:", response) # The answer is salmon