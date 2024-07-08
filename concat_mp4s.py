# concatenates all mp4s into one

# create video_list.txt

from glob import glob
import os
import subprocess
from tqdm import tqdm

video_dirs = sorted(glob("video_files/*"))


def run_command(command):
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0

start_idx = 10 # start at last interrupt 
for dir_ in tqdm(video_dirs[start_idx:]): 
    os.chdir(dir_)
    if os.path.exists("combined.mp4"):
        result = subprocess.run(['rm', 'combined.mp4']) # remove if it already exists
    mp4s = sorted(glob("*.mp4"))
    #print(mp4s)
    
    with open("video_list.txt", "w") as f:
        for mp4 in mp4s:
            #mp4 = mp4.split("/")[-1]
            string = "file " + mp4 + "\n"
            #print(string)
            f.write(string)
            #f.write("file " + mp4 + "\n")
    
    # Read the video list from the video_list.txt file
    input_videos = []
    with open('video_list.txt', 'r') as file:
        for line in file:
            if line.startswith('file'):
                video_path = line.split(' ')[1].split('\n')[0]  # Extract the video file path
                input_videos.append(video_path)

    encoded_videos = []


    # Re-encode each video individually
    for i, video in enumerate(input_videos):
        output_video = f"encoded_{i}.mp4"
        encoded_videos.append(output_video)
        command = [
            'ffmpeg', '-y', '-i', video, '-c:v', 'libx264', '-c:a', 'aac', '-strict', 'experimental',
            '-preset', 'fast', '-movflags', 'faststart', '-r', '30', '-vsync', '1', output_video
        ]

        if not run_command(command):
            print(f"In directory {os.getcwd()}")
            print(f"Directory contents: {sorted(os.listdir())}")
            print(f"Error re-encoding {video}")
            exit()
    
    # If all videos were re-encoded successfully, proceed to concatenation
    if len(encoded_videos) == len(input_videos):
        with open('encoded_video_list.txt', 'w') as f:
            for video in encoded_videos:
                f.write(f"file '{video}'\n")

        # Concatenate re-encoded videos
        command_concat = [
            'ffmpeg', '-f', 'concat', '-safe', '0', '-i', 'encoded_video_list.txt',
            '-fflags', '+genpts', '-c', 'copy', 'combined.mp4'
        ]
        if not run_command(command_concat):
            print("Error concatenating re-encoded videos! Video dir was: ", dir_)

    # Clean up encoded videos if necessary
    for video in encoded_videos:
        os.remove(video)

    os.chdir("../../")





