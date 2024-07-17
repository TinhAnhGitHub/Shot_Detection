import os
import torch
import numpy as np
import tqdm
from typing import Dict
from torchinfo import summary
from .AutoShot.utils import get_batches, get_frames
from transnetv2 import TransNetV2
from .AutoShot.model import AutoShot
#
class ShotDetection:
    """
    A class for performing shot detection on videos using either AutoShot or TransNetV2 models.
    """

    def __init__(self, choice: str = 'autoshot'):
        """
        Initialize the ShotDetection class.

        Args:
            choice (str): The model to use for shot detection. Either 'autoshot' or 'transnetv2'.
        """
        self.choice = choice.lower()
        if self.choice == 'autoshot':
            self.model = AutoShot("./AutoShot/model_weight/ckpt_0_200_0.pth")
        elif self.choice == 'transnetv2':
            self.model = TransNetV2()
        else:
            raise ValueError("Invalid choice. Please choose 'autoshot' or 'transnetv2'.")

    def run_model(self, video_path_dict: Dict[str, str]) -> Dict[str, np.ndarray]:
        """
        Run shot detection on a dictionary of video paths.

        Args:
            video_path_dict (Dict[str, str]): A dictionary mapping video names to their file paths.

        Returns:
            Dict[str, np.ndarray]: A dictionary mapping video names to their detected scene boundaries.
        """
        if self.choice == 'autoshot':
            return self._run_autoshot(video_path_dict)
        else:
            return self._run_transnetv2(video_path_dict)

    def _run_autoshot(self, video_path_dict: Dict[str, str]) -> Dict[str, np.ndarray]:
        """
        Run shot detection using the AutoShot model.

        Args:
            video_path_dict (Dict[str, str]): A dictionary mapping video names to their file paths.

        Returns:
            Dict[str, np.ndarray]: A dictionary mapping video names to their detected scene boundaries.
        """
        results = self.model.process_videos(video_path_dict)
        prediction_scenes = {}
        for video_name, predictions in results.items():
            prediction_scenes[video_name] = self.model.predictions_to_scenes(predictions)
        return prediction_scenes

    def _run_transnetv2(self, video_path_dict: Dict[str, str]) -> Dict[str, np.ndarray]:
        """
        Run shot detection using the TransNetV2 model.

        Args:
            video_path_dict (Dict[str, str]): A dictionary mapping video names to their file paths.

        Returns:
            Dict[str, np.ndarray]: A dictionary mapping video names to their detected scene boundaries.
        """
        result = {}
        for fnm, video_path in tqdm.tqdm(video_path_dict.items(), desc="Processing videos"):
            _, single_frame_predictions, _ = self.model.predict_video(video_path)
            scenes = self.model.predictions_to_scenes(single_frame_predictions)
            result[fnm] = scenes
        return result

    @staticmethod
    def get_model_summary(model: torch.nn.Module, input_size: tuple = (1, 3, 27, 48)) -> str:
        """
        Generate a summary of the model architecture using torchinfo.

        Args:
            model (torch.nn.Module): The model to summarize.
            input_size (tuple): The input size for the model. Default is (1, 3, 27, 48).

        Returns:
            str: A string representation of the model summary.
        """
        return str(summary(model, input_size=input_size, verbose=0))