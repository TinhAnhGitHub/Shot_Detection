import os
from typing import Dict, Any
from AutoShot.model import AutoShot
from AutoShot.keyframe_extractor import KeyFrameExtractor
from drive_loader import DriveUploader
from tqdm import tqdm


class VideoProcessor:
    def __init__(self, pretrained_model_path: str, keyframe_dir: str, credentials_path: str, token_path: str, drive_folder_id: str,  user_service_account = True):
        self.shot_detector = AutoShot(pretrained_model_path)
        self.keyframe_extractor = KeyFrameExtractor(keyframe_dir)
        #### DEBUG 
        # if user_service_account:

        #     self.drive_uploader = DriveUploader(
        #             credentials_path=credentials_path,
        #             use_service_account=True
        #         ) 
        # else:
        #     self.drive_uploader = DriveUploader(
        #         credentials_path=credentials_path,
        #         token_path=token_path
        #     )
        # self.drive_folder_id = drive_folder_id

    def process_videos(self, video_dict: Dict[str, Any]) -> None:
        total_videos = sum(1 for _ in self._flatten_dict(video_dict))
        
        print("\n----------------")
        print(f"Starting to process {total_videos} videos")
        print("----------------\n")

        with tqdm(total=total_videos, desc="\nOverall Progress", unit="video") as pbar:

            def process_nested(nested_dict: Dict[str, Any], current_path: str = ""):
                for key, value in nested_dict.items():
                    new_path = os.path.join(current_path, key)
                    if isinstance(value, str):
                        print(f"\nProcessing: {new_path}")
                        print("----------------")
                        try:
                            scenes = self.shot_detector.process_video(value)
                            
                            if scenes:
                                print(f"Detected {len(scenes)} scenes in {new_path}")
                                try:
                                    video_keyframe_dir = os.path.join(self.keyframe_extractor.keyframe_dir, new_path)
                                    self.keyframe_extractor.extract_keyframes(value, scenes, new_path)
                                    print(f"Finished extracting keyframes for {new_path}")

                                    # Create a folder in Google Drive for this video
                                    drive_folder_id = self.drive_uploader.create_folder_tree(self.drive_folder_id, new_path)

                                    # Upload keyframes to Google Drive
                                    self.drive_uploader.upload_folder(video_keyframe_dir, drive_folder_id)
                                    print(f"Uploaded keyframes for {new_path} to Google Drive")

                                except RuntimeError as e:
                                    print(f"Error extracting keyframes from {new_path}: {str(e)}")
                            else:
                                print(f"No scenes detected in video: {new_path}")
                        except FileNotFoundError as e:
                            print(f"File not found: {str(e)}")
                        except ValueError as e:
                            print(f"Error processing video {new_path}: {str(e)}")
                        except RuntimeError as e:
                            print(f"Runtime error processing video {new_path}: {str(e)}")
                        except Exception as e:
                            print(f"Unexpected error processing video {new_path}: {str(e)}")
                        finally:
                            pbar.update(1)
                            print("----------------\n")
                    elif isinstance(value, dict):
                        process_nested(value, new_path)
                    else:
                        print(f"Unexpected item in video_dict: {new_path}")

            process_nested(video_dict)

    def _flatten_dict(self, d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    


    