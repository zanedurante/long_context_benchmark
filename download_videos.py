# given a list of mp4 files from the web, download locally

import os
import requests
from tqdm import tqdm

PROBLEM_VIDEOS = [
    "1_stock-footage-beef-fillet-slices-being-seasoned-with-pepper-and-salt.mp4"
]

NUM_SUBVIDEOS = 10
CATEGORY="action"



OUTPUT_DIR = f'video_files_{NUM_SUBVIDEOS}'

def download_videos(file_paths, output_dir='video_files'):
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, file_path in enumerate(file_paths):
        file_name = file_path.split('/')[-1]
        # prepend idx to maintain order (prepend with 0s to maintain order in file system)
        str_idx = str(idx).zfill(4)
        file_name = str_idx + "_" + file_name
        #print("Downloading:", file_name)
        if file_name in PROBLEM_VIDEOS:
            print(f"Skipping {file_name} due to issues with the video")
            continue
        if os.path.exists(f'{output_dir}/' + file_name):
            # File already exists
            continue
        r = requests.get(file_path, allow_redirects=True)
        open(f'{output_dir}/' + file_name, 'wb').write(r.content)


# Load file_paths from the video_questions file
with open(f'video_questions/video_questions_{CATEGORY}_n={NUM_SUBVIDEOS}.txt', 'r', errors="ignore") as f:
    video_questions = eval(f.read())
#    import pdb; pdb.set_trace()

# get the file_paths
for idx in tqdm(range(len(video_questions))):
    output_dir = OUTPUT_DIR + "/" + str(idx) + "/"
    os.makedirs(output_dir, exist_ok=True)
    file_paths = video_questions[idx]['file_paths']
    download_videos(file_paths, output_dir=output_dir)
    # also write the questions and answers to separate files
    with open(output_dir + f'{CATEGORY}_question.txt', 'w') as f:
        f.write(str(video_questions[idx]['question']) + '\n')
    with open(output_dir + f'{CATEGORY}_answer.txt', 'w') as f:
        f.write(str(video_questions[idx]['answer']) + '\n')