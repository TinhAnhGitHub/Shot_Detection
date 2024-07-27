import os
from typing import Dict, Any, Iterator, Deque
from collections import deque
from AutoShot.model import AutoShot
from AutoShot.keyframe_extractor import KeyFrameExtractor
from tqdm import tqdm

class VideoProcessor:
    def __init__(self, pretrained_model_path: str, keyframe_dir: str):
        self.shot_detector = AutoShot(pretrained_model_path)
        self.keyframe_extractor = KeyFrameExtractor(keyframe_dir)

    def _bfs_get_video_paths(self, input_dir: str) -> Iterator[str]:
        
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
        queue: Deque[str] = deque([input_dir])

        while queue:
            current_dir = queue.popleft()
            with os.scandir(current_dir) as entries:
                for entry in entries:
                    if entry.is_dir():
                        queue.append(entry.path)
                    elif entry.is_file() and entry.name.lower().endswith(video_extensions):
                        yield entry.path
    
    def _process_single_video(self, *, video_path: str, relative_path: str) -> None:
        try:
            scenes = self.shot_detector.process_video(video_path=video_path)
            if scenes:
                print(f"Detected {len(scenes)} scenes in {relative_path}")
                video_keyframe_dir = os.path.join(self.keyframe_extractor.keyframe_dir, os.path.dirname(relative_path))
                self.keyframe_extractor.extract_keyframes(video_path, scenes, relative_path)
                print(f"Finished extracting keyframes for {relative_path}")
                print(f"Keyframes saved in: {video_keyframe_dir}")
            else:
                print(f"No scenes detected in video: {relative_path}")
        except FileNotFoundError as e:
            print(f"File not found: {str(e)}")
        except ValueError as e:
            print(f"Error processing video: {str(e)}")
        except RuntimeError as e:
            print(f"Runtime error: {str(e)}")
        
    
    def process_videos(self, input_dir: str)-> None:
        
        video_paths = list(self._bfs_get_video_paths(input_dir))
        total_videos = len(video_paths)


        print("\n----------------")
        print(f"Starting to process {total_videos} videos")
        print("----------------\n")

        for video_path in tqdm(video_paths, desc="Overall Progress", unit="video"):
            relative_path = os.path.relpath(video_path, input_dir)
            print(f"\nProcessing: {relative_path}")
            print("---------------- ")

            try:
                self._process_single_video(video_path= video_path, relative_path= relative_path)

            except Exception as e:
                 print(f"Error processing video {relative_path}: {str(e)}")
            print("----------------\n")


    