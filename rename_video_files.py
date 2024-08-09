video_dir = "video_files_10"

# Go through each subdir and rename the mp4 files in that subdir from x_filename.mp4 to be 000x_filename.mp4 (zero filled)

import os
from tqdm import tqdm

for subdir in tqdm(os.listdir(video_dir)):
    subdir_path = os.path.join(video_dir, subdir)
    if os.path.isdir(subdir_path):
        for file in os.listdir(subdir_path):
            if file.endswith(".mp4"):
                new_file = file.split("_")[0].zfill(4) + "_" + file.split("_")[1]
                os.rename(os.path.join(subdir_path, file), os.path.join(subdir_path, new_file))
