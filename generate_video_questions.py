import pandas as pd
import openai 
import os
import time

with open('keys/openai.key', 'r') as f:
    openai.api_key = f.readline().strip()

def call_gpt(prompt):
    response = openai.chat.completions.create(
        model="gpt-4",
        temperature=0.9,
        top_p=0.3,
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
Generate a question and answer pair that would be answerable after sequentially watching videos with the following captions.  The person watching the videos doesn't see the captions or hear any audio, only the visual content in the videos.  Ensure your questions do not require jargon or domain-specific answers.  Do not ask simply about the order of events and remember that each caption is on a new line and represents a different video clip.
CAPTIONS 
Generate it in the format: {'question': '...', 'answer': '...'}."""

# Load the CSV files
results_df = pd.read_csv('rewrite_10M_val.csv')
num_videos_per_batch = 20

# load in the first 5 video captions
start_idx=0 # 4030 # TODO: Run with this value!
current_idx = start_idx #0
num_samples_processed = start_idx / 10
NUM_TOTAL = 4770 #1000
all_questions = []

def get_video_questions(captions):
    captions_string = ""
    for caption in captions:
        captions_string += caption + "\n"

    prompt = base_prompt.replace("CAPTIONS", captions_string)
    
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
        questions = eval(get_video_questions(captions))
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

with open(f'video_questions/video_questions_n={num_videos_per_batch}-start={int(start_idx/10)}.txt', 'w') as f:
    f.write(str(all_questions) + '\n')