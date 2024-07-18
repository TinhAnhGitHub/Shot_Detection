from .supernet import TransNetV2Supernet
import os
import torch
import numpy as np
from typing import List, Optional
from .utils import get_batches, get_frames
from tqdm import tqdm


class AutoShot:
    """
        A class for automatic shot detection in video using TransNetV2Supernet model
    """

    def __init__(
        self,
        pretrained_path: str,
        device: Optional[str] = None
    ):
        """Initialize the Autoshot class

        Args:
            pretrained_path (str): Path to the pretrained model
            keyframe_output_dir (str): Directory to save the keyframes
            device (Optional[str], optional): Device to run the model 'cpu' or 'cuda'. Defaults to None.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(pretrained_path=pretrained_path)
    
    def _load_model(self, pretrained_path: str) -> torch.nn.Module:
        """Loading the pretrained model

        Args:
            pretrained_path (str): Path to the pretrained model weights

        Raises:
            FileNotFoundError: The pretrained file is missing
            RuntimeError: If loading model process is not successful

        Returns:
            torch.nn.Module : loaded and configured model
        """
        try:
            model =  TransNetV2Supernet().eval()
            if not os.path.exists(pretrained_path):
                raise FileNotFoundError(f"Can't find the pretrained model path at {pretrained_path}")
            
            print(f"Loading the pretrained model from {pretrained_path}")
            model_dict = model.state_dict()
            pretrained_dict = torch.load(pretrained_path, map_location=self.device)
            pretrained_dict = {k: v for k, v in pretrained_dict['net'].items() if k in model_dict}
            print(f"Current model has {len(model_dict)} params, Updating {len(pretrained_dict)} params")
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            return model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load the model. Did you piss off the AI gods? Error: {str(e)}")
        
        
    
            
    def predict(self, batch: np.ndarray) -> np.ndarray:
        """Make predictions on the batch of frames

        Args:
            batch (np.ndarray): Batch of video frames, in the shape of (height, width, color_channel, frames)
            typically: (27, 48, channels=3, frames = 100)

        Returns:
            np.ndarray: Predictions of the batch
        """
        with torch.no_grad():
            batch = torch.from_numpy(batch.transpose((3, 0, 1, 2))[np.newaxis, ...]) * 1.0
            batch = batch.to(self.device)
            one_hot = self.model(batch)

            if isinstance(one_hot, tuple):
                one_hot = one_hot[0]
            return torch.sigmoid(one_hot[0]).cpu().numpy()
    
    def detect_shots(self, frames: np.ndarray) -> np.ndarray:
        """Detects shot in a video

        Args:
            frames (np.ndarray): Array of video frames, (num_frames, height, width, channels)

        Returns:
            np.ndarray: shot detection predictions for each frame
        """
        predictions= []
        for batch in tqdm(get_batches(frames=frames), desc="Dectecting shots", unit="batch"):
            prediction = self.predict(batch=batch)
            predictions.append(prediction[25:75])
        
        return np.concatenate(predictions, axis=0)[:len(frames)]
    
    @staticmethod 
    def predictions_to_scenes(predictions: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Convert frame-wise predictions to scene boundaries.

        Args:
            predictions (np.ndarray): Array of frame-wise predictions
            threshold (float, optional): Threshold for considering a frame as a shot boundary. Defaults to 0.5

        Returns:
            np.ndarray: List of scene start and end frame indices
        """
        predictions = (predictions > threshold).astype(np.uint8)
        scenes = []
        t, t_prev, start = -1, 0, 0
        for i, t in enumerate(predictions):
            if t_prev == 1 and t == 0:
                start = i
            if t_prev == 0 and t == 1 and i != 0:
                scenes.append([start, i])
            t_prev = t
        if t == 0:
            scenes.append([start, i])

        # just fix if all predictions are 1
        if len(scenes) == 0:
            return np.array([[0, len(predictions) - 1]], dtype=np.int32)
        return np.array(scenes, dtype=np.int32)

    def process_video(self, video_path: str) -> List[List[int]]:
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"File not found: {video_path}")

            frames = get_frames(video_file_path=video_path)
            if frames is None or len(frames) == 0:
                raise ValueError(f"No frames extracted from video: {video_path}")
            
            predictions = self.detect_shots(frames = frames)
            scenes = self.predictions_to_scenes(predictions=predictions)

            return scenes.tolist()

        except Exception as e:
            raise RuntimeError(F"Failed to process video: {video_path}. Error: {e}")
        