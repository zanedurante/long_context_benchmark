import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm
from PIL import Image
from moviepy.editor import ImageSequenceClip, ImageClip
from video_transitions import apply_transitions



# get number of frames + duration in seconds
def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return 0, 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps
    cap.release()
    return frame_count, duration

# get frame in video at time t (sec)
def get_frame(video_path, t):
    if t < 0:
        raise ValueError("Time t must be greater than or equal to 0")
    
    cap = cv2.VideoCapture(video_path)
    num_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)


    if t > num_video_frames / fps + 1e-6: # add epsilon to account for floating point errors
        raise ValueError(f"Time t={t} must be less than or equal to video duration {num_video_frames / fps} seconds")

    frame_num = int(t * fps)
    if frame_num >= num_video_frames:
        frame_num -= num_video_frames - 1 # OpenCV doesn't like reading the last frame of a video

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    try:
        ret, frame = cap.read()
    except:
        import pdb; pdb.set_trace()
    if not ret:
        raise ValueError(f"Error reading frame for video {video_path} at time {t} seconds")
    
    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def get_total_time_for_videos(video_paths):
    total_time = 0
    for video_path in video_paths:
        frame_count, duration = get_video_info(video_path)
        total_time += duration
    return total_time

def get_frame_times(total_time, num_frames):
    frame_times = np.linspace(0, total_time, num_frames)
    return frame_times

def calculate_frame_times_across_videos(video_paths, frame_times):
    # returns video_path, frame_time pairs for each frame_time.  
    # We calculate video-specific frame times by accounting for the fact that each video is played in sequence.

    video_frame_times = []
    final_video_paths = []

    current_time = 0
    for video in video_paths:
        frame_count, duration = get_video_info(video)
        time_at_end_of_video = current_time + duration

        # check how many frame_times are in this video
        for t in frame_times:
            if t >= current_time and t <= time_at_end_of_video:
                video_frame_times.append(t - current_time)
                final_video_paths.append(video)
        
        current_time = time_at_end_of_video
    
    return final_video_paths, video_frame_times     

def get_frames_across_video_list(video_paths, num_frames, transition_type='base', num_transition_frames=10):
    total_time = get_total_time_for_videos(video_paths)
    frame_times = get_frame_times(total_time, num_frames)
    video_paths, frame_times = calculate_frame_times_across_videos(video_paths, frame_times)
    frames = []
    descriptions = []

    for video_path, frame_time in zip(video_paths, frame_times):
        frame = get_frame(video_path, frame_time)
        frames.append(frame)

        # Extract description from video path
        if "_stock-" in video_path:
            description = video_path.split("_stock-")[-1].split(".")[0]  # Extract description
            descriptions.append(description)
        else:
            descriptions.append("default_description")  # Fallback if no description found

    if transition_type != 'base':
        frames = apply_transitions(frames, video_paths, transition_type, num_transition_frames, descriptions)
    
    return frames


def create_video_from_images(image_files, output_path, fps=1):
    # Ensure there is at least one image file
    if not image_files:
        raise ValueError("No image files provided")

    # Get the size of the first image
    first_image = Image.open(image_files[0])
    first_image_size = first_image.size

    # Resize all images to the size of the first image
    resized_images = []
    for image_file in image_files:
        image = Image.open(image_file)
        resized_image = image.resize(first_image_size, Image.LANCZOS)
        resized_image_path = f"resized_{os.path.basename(image_file)}"
        resized_image.save(resized_image_path)
        resized_images.append(resized_image_path)

    # Create a video clip from the resized images
    clip = ImageSequenceClip(resized_images, fps=fps)

    # Write the video to a file
    clip.write_videofile(output_path, codec='libx264')

    # Clean up resized images
    for image_file in resized_images:
        os.remove(image_file)

if __name__ == "__main__":
    # Add argument parsing
    parser = argparse.ArgumentParser(description='Extract frames from video directories')
    parser.add_argument('--video_dir', type=str, default="video_files_10", 
                        help='Directory containing video subdirectories')
    parser.add_argument('--num_frames', type=int, default=32, 
                        help='Number of frames to extract')
    parser.add_argument('--transition_type', type=str, default='base', 
                        choices=['base', 'fade', 'slide_right', 'slide_left', 'diffusion'],
                        help='Type of transition between frames')
    parser.add_argument('--num_transition_frames', type=int, default=10, 
                        help='Number of frames for transitions')
    parser.add_argument('--output_fps', type=int, default=4, 
                        help='FPS for the output video')
    
    # Parse arguments
    args = parser.parse_args()

    # Get video directories
    video_dirs = os.listdir(args.video_dir)
    video_dirs = sorted([os.path.join(args.video_dir, d) for d in video_dirs])[:1]
    print(video_dirs)
    
    for video_dir in tqdm(video_dirs):
        video_paths = sorted([os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")])
        frames = get_frames_across_video_list(video_paths, args.num_frames, 
                                              transition_type=args.transition_type, 
                                              num_transition_frames=args.num_transition_frames)
        
        # Create a folder with transition type
        transition_folder = os.path.join(video_dir, f"{args.num_frames}_frames_{args.transition_type}")
        os.makedirs(transition_folder, exist_ok=True)

        for idx, frame in enumerate(frames):
            idx_str = str(idx).zfill(4)
            cv2.imwrite(os.path.join(transition_folder, f"frame_{idx_str}.jpg"), 
                        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        # Create a video from the frames
        image_files = [os.path.join(transition_folder, f) for f in os.listdir(transition_folder) if f.endswith(".jpg")]
        image_files = sorted(image_files)
        output_path = os.path.join(transition_folder, "combined.mp4")
        create_video_from_images(image_files, output_path, fps=args.output_fps)

        
