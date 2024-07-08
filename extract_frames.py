import cv2
import numpy as np
import os
from tqdm import tqdm


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

def get_frames_across_video_list(video_paths, num_frames):
    total_time = get_total_time_for_videos(video_paths)
    # TODO: Change the following line
    frame_times = get_frame_times(total_time, num_frames)
    video_paths, frame_times = calculate_frame_times_across_videos(video_paths, frame_times)
    frames = []
    for video_path, frame_time in zip(video_paths, frame_times):
            frame = get_frame(video_path, frame_time)
            frames.append(frame)
    return frames


if __name__ == "__main__":
    video_dir = "video_files"
    video_dirs = os.listdir(video_dir)
    video_dirs = [os.path.join(video_dir, d) for d in video_dirs]

    # for each video_dir, get all the mp4 videos
    NUM_FRAMES = 64
    for video_dir in tqdm(video_dirs):
        video_paths = sorted([os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")])
        frames = get_frames_across_video_list(video_paths, NUM_FRAMES)
        for idx, frame in enumerate(frames):
            os.makedirs(os.path.join(video_dir, f"{NUM_FRAMES}_frames"), exist_ok=True)
            idx_str = str(idx).zfill(4)
            cv2.imwrite(f"{video_dir}/{NUM_FRAMES}_frames/frame_{idx_str}.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
