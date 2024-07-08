# Given idx, go to video_files/idx/16_frames and visualize as 4x4 grid and save to video_files/idx/16_frames.png

from PIL import Image
import os
from tqdm import tqdm


def load_image(path):
    return Image.open(path)


def concat_frames_into_one_image(frames):
    # Concatenate frames into a 4x4 grid
    width, height = frames[0].size
    total_width = 4 * width
    total_height = 4 * height
    new_im = Image.new('RGB', (total_width, total_height))
    for idx, frame in enumerate(frames):
        x = idx % 4
        y = idx // 4
        new_im.paste(frame, (x * width, y * height))
    return new_im

def visualize_frames(idx):
    video_dir = "video_files"
    output_dir = os.path.join(video_dir, str(idx))
    video_dir = os.path.join(video_dir, str(idx), "16_frames")
    frame_paths = [os.path.join(video_dir, f) for f in sorted(os.listdir(video_dir))]
    frames = [load_image(f) for f in frame_paths]
    image = concat_frames_into_one_image(frames)
    image.save(os.path.join(output_dir, "16_frames.png"))
    return image

if __name__ == "__main__":
    #visualize_frames(0)
    for idx in tqdm(range(400)):
        visualize_frames(idx)
