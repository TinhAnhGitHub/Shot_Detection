import numpy as np
import ffmpeg

def get_frames(video_file_path: str, width: int = 48, height: int = 27) -> np.ndarray:
    """
    Extract frames from video like you're performing a magic trick, but with more swearing.
    
    Args:
        video_file_path (str): Path to the video file. Don't fuck this up.
        width (int): Width of the extracted frame. Default is 48, because we're not made of pixels.
        height (int): Height of the extracted frames. Default is 27, because odd numbers are cool.
    
    Returns:
        np.ndarray: Array of video frames. If this fails, you're proper fucked.
    """
    try:
        probe = ffmpeg.probe(video_file_path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        _ = int(video_info['nb_frames'])
        
        out, _ = (
            ffmpeg
            .input(video_file_path)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}')
            .run(capture_stdout=True, capture_stderr=True)
        )
        
        video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
        return video
    except ffmpeg.Error as e:
        print(f"ffmpeg error: {e.stderr.decode()}")
        raise
    except Exception as e:
        print(f"Error in get_frames: {str(e)}")
        raise

def get_batches(frames: np.ndarray):
    """
    Prepare batches of frames for processing. It's like making a video sandwich.
    
    Args:
        frames (np.ndarray): Array of video frames. Try not to feed it pictures of your ex.
    
    Yields:
        np.ndarray: Batches of frames, because processing all at once would make your computer cry.
    """
    reminder = 50 - len(frames) % 50
    if reminder == 50:
        reminder = 0
    frames = np.concatenate([frames[:1]] * 25 + [frames] + [frames[-1:]] * (reminder + 25), 0)

    
    for i in range(0, len(frames) - 50, 50):
        yield frames[i:i + 100]