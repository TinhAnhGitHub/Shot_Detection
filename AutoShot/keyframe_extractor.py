import os
import cv2
import numpy as np
from typing import List 



class KeyFrameExtractor:
    def __init__(self, keyframe_dir: str):
        self.keyframe_dir = keyframe_dir
        os.makedirs(self.keyframe_dir, exist_ok=True)

    def sample_frames_from_shot(self, start: int, end: int, num_samples: int = 3) -> List[int]:
        return [start + i * (end - start) // (num_samples - 1) for i in range(num_samples)]

    def save_frame(self, frame: np.ndarray, filename: str) -> bool:
        return cv2.imwrite(filename, frame)
    
    def extract_keyframes(self, video_path: str, scenes: List[List[int]], output_prefix: str) -> None:
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")

            cap = cv2.VideoCapture(video_path)
            last_folder = os.path.basename(output_prefix)

            for _, (start, end) in enumerate(scenes):
                sample_frames = self.sample_frames_from_shot(start, end)
                for _, frame_idx in enumerate(sample_frames):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
                    ret, frame = cap.read()
                    if ret:
                        video_keyframe_dir = os.path.join(self.keyframe_dir, output_prefix)
                        os.makedirs(video_keyframe_dir, exist_ok=True)
                        
                        keyframe_filename = f"video_{last_folder}_index_{frame_idx}.jpg"
                        keyframe_path = os.path.join(video_keyframe_dir, keyframe_filename)
                        
                        if not self.save_frame(frame=frame, filename=keyframe_path):
                             print(f"Failed to save frame {frame_idx} for video {output_prefix}")
                    else:
                        print(f"Failed to read frame {frame_idx} for video {output_prefix}")
            
            cap.release()
        except Exception as e:
            raise RuntimeError(f"Failed to extract keyframes from video {video_path}. Error: {str(e)}")
                    
        