import pandas as pd
import openai 
import os
import time
from gpt4v_video import GLOBAL_TEMPERATURE

with open('keys/openai.key', 'r') as f:
    openai.api_key = f.readline().strip()

def call_gpt(prompt):
    response = openai.chat.completions.create(
        model="gpt-4o",
        temperature=GLOBAL_TEMPERATURE,
        messages=[
            {
                "role": "system", "content": "You are a helpful assistant.",
                "role": "user", "content": prompt
            }
        ]
    )
    #import pdb; pdb.set_trace()
    return response.choices[0].message.content, response.usage.total_tokens


base_prompt = """
Generate a question and answer pair that would be answerable after sequentially watching videos with the following captions.  The person watching the videos doesn't see the captions or hear any audio, only the visual content in the videos.  Ensure your questions do not require jargon or domain-specific answers.  Remember that each caption is on a new line and represents a different video clip.
DO NOT use specific locations in your answer even if this information is present in the captions.
CATEGORY
CAPTIONS 
Generate it in the format: {'question': '...', 'answer': '...'}."""

# Load the CSV files
results_df = pd.read_csv('rewrite_10M_val.csv')
num_videos_per_batch = 10
CATEGORY_USED = "action"
# load in the first 5 video captions
start_idx=0 # 4030 
current_idx = start_idx #0
num_samples_processed = start_idx / 10
NUM_TOTAL = 100 #4770 #1000
all_questions = []

print("Generating questions for the category:", CATEGORY_USED)

def get_video_questions(captions, category="general"):
    captions_string = ""
    for caption in captions:
        captions_string += caption + "\n"

    prompt = base_prompt.replace("CAPTIONS", captions_string)

    if category == "general":
        prompt = prompt.replace("CATEGORY", "")
    elif category == "action":
        action_prompt = "Create a specific question focused on the activities that people are doing in the videos.  Be sure to ask specific questions about specific video clips."
        prompt = prompt.replace("CATEGORY", action_prompt)
    elif category == "scene":
        location_prompt = "Create a specific question focused on the scenes and locations present in the videos."
        prompt = prompt.replace("CATEGORY", location_prompt)
    elif category == "object":
        object_prompt = "Create a specific question focusing on the important objects that appear in the videos. Do not simply ask to list the key objects, but pose a question that requires reasoning about the objects' role in the videos."
        prompt = prompt.replace("CATEGORY", object_prompt)
    elif category == "temporal":
        temporal_prompt = "Create a specific question focusing on the order of events in the videos and the temporal relationships between them. Avoid asking basic before/after questions. A good question would require seeing 2-3 scenes and events across the entire set of videos."
        prompt = prompt.replace("CATEGORY", temporal_prompt)
    else:
        raise ValueError(f"Invalid category for question generation. Please choose from 'general', 'action', 'scene', or 'object'. Category given was: {category}.")
    
    
    questions, usage = call_gpt(prompt)

    return questions

num_repeat_errors = 0 

while current_idx < len(results_df) and num_samples_processed < NUM_TOTAL:
    # get the next 5 video captions
    print("Processing idx:", current_idx, "num_samples_processed:", num_samples_processed, "total_samples:", min(NUM_TOTAL, len(results_df)))
    batch_df = results_df[current_idx:current_idx+num_videos_per_batch]
    current_idx += num_videos_per_batch

    # get the 'name' column as list
    captions = batch_df['name'].tolist()
    paths = batch_df['contentUrl'].tolist()
        
    # print the captions
    try:
        questions = eval(get_video_questions(captions, category=CATEGORY_USED))
    except KeyboardInterrupt:
        break
    except:
        print("Error in generating questions for idx:", current_idx, "trying again!")
        current_idx -= num_videos_per_batch
        num_repeat_errors += 1
        if num_repeat_errors > 1:
            time.sleep(1)
        if num_repeat_errors > 5:
            time.sleep(10)
        elif num_repeat_errors > 10:
            time.sleep(60)
        elif num_repeat_errors > 20:
            time.sleep(180)
        continue
    num_repeat_errors = 0
    questions['file_paths'] = paths
    all_questions.append(questions)



    num_samples_processed += 1

    #import pdb; pdb.set_trace()

os.makedirs('video_questions', exist_ok=True)

save_location = f'video_questions/video_questions_{CATEGORY_USED}_n={num_videos_per_batch}-start={int(start_idx/10)}.txt'
print("Saving to:", save_location)
with open(save_location, 'w') as f:
    f.write(str(all_questions) + '\n')

for q in all_questions:
    print(q['question'])