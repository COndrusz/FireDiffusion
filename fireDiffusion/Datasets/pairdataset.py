"""
Christopher Ondrusz
GitHub: acse_cro23
"""
import torch
from torch.utils.data import Dataset
import cv2
import os


class VideoFramePairsDataset(Dataset):
    """
    A dataset class for creating pairs of grayscale video frames from a
    directory of video files.

    Attributes:
    -----------
    video_dir : str
        The directory containing the video files.
    transform : callable, optional
        A function/transform to apply to the video frames.
    timegap : int, optional
        The gap between consecutive frames in a pair, default is 1.
    video_files : list of str
        A list of paths to the video files in the directory.
    frame_pairs : list of tuples
        A list of tuples, where each tuple contains two consecutive frames
        with a time gap.
    """

    def __init__(self, video_dir, transform=None, timegap=1):
        """
        Initializes the VideoFramePairsDataset.

        Parameters:
        -----------
        video_dir : str
            The directory containing the video files.
        transform : callable, optional
            A function/transform to apply to the video frames.
        timegap : int, optional
            The gap between consecutive frames in a pair, default is 1.
        """
        self.video_dir = video_dir
        self.transform = transform
        self.timegap = timegap
        self.video_files = [os.path.join(video_dir, f)
                            for f in os.listdir(video_dir)
                            if f.endswith('.mp4')]
        self.frame_pairs = []
        self._prepare_dataset()

    def _prepare_dataset(self):
        """
        Prepares the dataset by extracting grayscale frame pairs from the
        video files.

        This method reads each video file, converts each frame to grayscale,
        and stores pairs of consecutive frames separated by the specified time
        gap.

        Returns:
        --------
        None
        """
        for video_file in self.video_files:
            cap = cv2.VideoCapture(video_file)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(frame_gray)
            cap.release()
            for i in range(len(frames) - self.timegap):
                self.frame_pairs.append((frames[i], frames[i + self.timegap]))

    def __len__(self):
        """
        Returns the total number of frame pairs in the dataset.

        Returns:
        --------
        int
            The total number of frame pairs in the dataset.
        """
        return len(self.frame_pairs)

    def __getitem__(self, idx):
        """
        Retrieves the frame pair at the specified index.

        Parameters:
        -----------
        idx : int
            The index of the frame pair to retrieve.

        Returns:
        --------
        tuple of torch.Tensor
            A tuple containing the first and second frames as grayscale
            tensors, and optionally transformed according to the specified
            transform.
        """
        frame, next_frame = self.frame_pairs[idx]
        frame = torch.from_numpy(frame).unsqueeze(0).float()
        next_frame = torch.from_numpy(next_frame).unsqueeze(0).float()

        if self.transform:
            frame = self.transform(frame)
            next_frame = self.transform(next_frame)

        return frame, next_frame
