import numpy as np
import cv2
from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
from diffusers import StableVideoDiffusionPipeline

class VideoTransition:
    def __init__(self):
        # Initialize Stable Diffusion pipeline for img2img
        self.pipe = None  # Lazy loading to save memory when not using diffusion
        
    def _init_diffusion(self):
        if self.pipe is None:
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16
            )
            if torch.cuda.is_available():
                self.pipe = self.pipe.to("cuda")
    
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

    def diffusion(self, frame1, frame2, alpha, strength=0.75, description=None):
        """
        Create a diffusion-based transition between frames
        
        Args:
            frame1 (np.ndarray): First frame
            frame2 (np.ndarray): Second frame
            alpha (float): Transition progress (0.0 to 1.0)
            strength (float): Strength of diffusion effect (0.0 to 1.0)
            description (str): Description of the transition
        """
        self._init_diffusion()
        
        # Resize frames
        height = min(frame1.shape[0], frame2.shape[0])
        width = min(frame1.shape[1], frame2.shape[1])
        frame1_resized = cv2.resize(frame1, (width, height))
        frame2_resized = cv2.resize(frame2, (width, height)) # Figure out how to use this
        
        # Create base transition using fade
        base_transition = self.fade(frame1_resized, frame2_resized, alpha)
        
        # Convert to PIL Image
        init_image = Image.fromarray(base_transition)
        
        # Process the description
        description = description.replace("footage", "").replace("-", " ").strip()
        
        # Generate diffusion transition
        output = self.pipe(
            prompt=description,
            image=init_image,
            strength=strength * alpha,
            guidance_scale=7.5,
            num_inference_steps=20
        ).images[0]
        
        # Convert back to numpy array
        return np.array(output)

def apply_transitions(frames, video_paths, transition_type='fade', num_transition_frames=10, descriptions=None):
    """
    Apply transitions between frames from different videos
    
    Args:
        frames (list): List of frames
        video_paths (list): List of video paths corresponding to each frame
        transition_type (str): Type of transition ('fade', 'slide_right', 'slide_left', 'diffusion')
        num_transition_frames (int): Number of frames for transition
    """
    if len(frames) < 2:
        return frames

    transition = VideoTransition()
    transitioned_frames = []
    
    for i in range(len(frames) - 1):
        transitioned_frames.append(frames[i])
        
        if i < len(frames) - 1 and video_paths[i] != video_paths[i + 1]:
            for j in range(num_transition_frames):
                alpha = (j + 1) / (num_transition_frames + 1)
                
                if transition_type == 'fade':
                    transition_frame = transition.fade(frames[i], frames[i+1], alpha)
                elif transition_type == 'slide_right':
                    transition_frame = transition.slide_horizontal(frames[i], frames[i+1], alpha, 'right')
                elif transition_type == 'slide_left':
                    transition_frame = transition.slide_horizontal(frames[i], frames[i+1], alpha, 'left')
                elif transition_type == 'diffusion':
                    transition_frame = transition.diffusion(frames[i], frames[i+1], alpha, 0.75, descriptions[i])
                else:
                    raise ValueError(f"Unknown transition type: {transition_type}")
                
                transitioned_frames.append(transition_frame)
    
    transitioned_frames.append(frames[-1])
    return transitioned_frames
