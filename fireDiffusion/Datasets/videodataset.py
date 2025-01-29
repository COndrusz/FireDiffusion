"""
Christopher Ondrusz
GitHub: acse_cro23
"""
import os
import cv2
import numpy as np
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    """
    A PyTorch Dataset class for loading frames from multiple videos stored in
    a folder. The frames are converted to grayscale and can be optionally
    transformed.

    Parameters:
    -----------
    video_folder : str
        The directory containing the video files.
    num_videos : int
        The number of videos to load from the folder.
    num_frames_per_video : int
        The number of frames to load from each video.
    frame_size : tuple
        The desired frame size (height, width) for the loaded frames.
    transform : callable, optional, default=None
        A function/transform that takes in a frame and returns a transformed
        version.

    Returns:
    --------
    None
    """
    def __init__(self, video_folder, num_videos, num_frames_per_video,
                 frame_size, transform=None):
        self.video_folder = video_folder
        self.num_videos = num_videos
        self.num_frames_per_video = num_frames_per_video
        self.frame_size = frame_size
        self.transform = transform
        self.videos = self.load_videos()

    def load_video(self, video_path):
        """
        Loads a single video from the specified path, converts its frames to
        grayscale, and resizes them to the specified frame size.

        Parameters:
        -----------
        video_path : str
            The path to the video file to be loaded.

        Returns:
        --------
        np.ndarray
            An array of grayscale frames from the video.
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray_frame)
        cap.release()
        return np.array(frames)

    def load_videos(self):
        """
        Loads frames from multiple videos in the specified folder. Each video
        is loaded, converted to grayscale, resized, and a specified number of
        frames are selected.

        Returns:
        --------
        list
            A list containing all the loaded and processed frames from all
            videos.
        """
        videos = []
        for i in range(self.num_videos):
            video_path = os.path.join(self.video_folder,
                                      f"fire_Chimney_video_{i+1}.mp4")
            video_frames = self.load_video(video_path)
            video_frames = video_frames[:self.num_frames_per_video]
            videos.extend(video_frames)
        return videos

    def __len__(self):
        """
        Returns the total number of frames in the dataset.

        Returns:
        --------
        int
            The total number of frames in the dataset.
        """
        return len(self.videos)

    def __getitem__(self, idx):
        """
        Retrieves a frame at the specified index and applies transformations
        if provided.

        Parameters:
        -----------
        idx : int
            The index of the frame to retrieve.

        Returns:
        --------
        torch.Tensor
            The processed frame as a torch.Tensor.
        """
        frame = self.videos[idx]
        if self.transform:
            frame = self.transform(frame)
        return frame
