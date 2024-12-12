import numpy as np
import cv2

class VideoTransition:
    @staticmethod
    def fade(frame1, frame2, alpha):
        """
        Create a fade transition between two frames
        
        Args:
            frame1 (np.ndarray): First frame
            frame2 (np.ndarray): Second frame
            alpha (float): Transition progress (0.0 to 1.0)
        
        Returns:
            np.ndarray: Blended frame
        """
        height = min(frame1.shape[0], frame2.shape[0])
        width = min(frame1.shape[1], frame2.shape[1])
        frame1_resized = cv2.resize(frame1, (width, height))
        frame2_resized = cv2.resize(frame2, (width, height))
        return cv2.addWeighted(frame1_resized, 1 - alpha, frame2_resized, alpha, 0)

    @staticmethod
    def slide_horizontal(frame1, frame2, alpha, direction='right'):
        """
        Create a horizontal slide transition
        
        Args:
            frame1 (np.ndarray): First frame
            frame2 (np.ndarray): Second frame
            alpha (float): Transition progress (0.0 to 1.0)
            direction (str): 'right' or 'left'
        
        Returns:
            np.ndarray: Transitioned frame
        """
        height = min(frame1.shape[0], frame2.shape[0])
        width = min(frame1.shape[1], frame2.shape[1])
        frame1_resized = cv2.resize(frame1, (width, height))
        frame2_resized = cv2.resize(frame2, (width, height))
        
        shift = int(width * alpha)
        
        transition_frame = np.zeros_like(frame1_resized)
        
        if direction == 'right':
            transition_frame[:, :width-shift] = frame1_resized[:, shift:]
            transition_frame[:, width-shift:] = frame2_resized[:, :shift]
        else:  # left
            transition_frame[:, shift:] = frame1_resized[:, :width-shift]
            transition_frame[:, :shift] = frame2_resized[:, width-shift:]
        
        return transition_frame

def apply_transitions(frames, video_paths, transition_type='fade', num_transition_frames=10):
    """
    Apply transitions between frames from different videos
    
    Args:
        frames (list): List of frames
        video_paths (list): List of video paths corresponding to each frame
        transition_type (str): Type of transition ('fade', 'slide_right', 'slide_left')
        num_transition_frames (int): Number of frames for transition
    
    Returns:
        list: Frames with transitions inserted
    """
    if len(frames) < 2:
        return frames

    transitioned_frames = []
    for i in range(len(frames) - 1):
        transitioned_frames.append(frames[i])
        
        if i < len(frames) - 1 and video_paths[i] != video_paths[i + 1]:
            for j in range(num_transition_frames):
                alpha = (j + 1) / (num_transition_frames + 1)
                
                if transition_type == 'fade':
                    transition_frame = VideoTransition.fade(frames[i], frames[i+1], alpha)
                elif transition_type == 'slide_right':
                    transition_frame = VideoTransition.slide_horizontal(frames[i], frames[i+1], alpha, 'right')
                elif transition_type == 'slide_left':
                    transition_frame = VideoTransition.slide_horizontal(frames[i], frames[i+1], alpha, 'left')
                else:
                    raise ValueError(f"Unknown transition type: {transition_type}")
                
                transitioned_frames.append(transition_frame)
    
    transitioned_frames.append(frames[-1])
    return transitioned_frames