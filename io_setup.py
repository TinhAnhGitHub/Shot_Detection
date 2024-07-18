import os
from typing import Dict

class DirectoryNotFoundError(Exception):
    """Custom exception for when the input directory is not found."""
    pass

def setup_video_path(input_dir: str) -> Dict[str, Dict]:
    def dfs(current_path: str, current_dict: Dict) -> None:
            for item in sorted(os.listdir(current_path)):
                item_path = os.path.join(current_path, item)
                if os.path.isdir(item_path):
                    current_dict[item] = {}
                    dfs(item_path, current_dict[item])
                elif item.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Add more video extensions if needed
                    video_id = os.path.splitext(item)[0]
                    current_dict[video_id] = item_path

    result = {}
    if not os.path.exists(input_dir):
        raise DirectoryNotFoundError(f"The input directory '{input_dir}' does not exist.")
    dfs(input_dir, result)
    return result
