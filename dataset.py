import os
import cv2
import re
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class ISLVideoDataset(Dataset):
    """
    Loads videos from a folder containing video files directly.
    Labels are extracted from filenames by removing leading numbers and cleaning.
    Expects:
        root_dir/
            video1.MOV
            video2.MOV
            ...
    """
    def __init__(self, root_dir, frames_per_video=8, img_size=(64, 64), augment=False):
        """
        Args:
            root_dir: path to folder with video files
            frames_per_video: number of frames to sample per video
            img_size: (height, width) tuple for resizing frames
            augment: whether to apply data augmentation
        """
        self.root_dir = root_dir
        self.frames_per_video = frames_per_video
        self.img_size = img_size
        self.augment = augment

        # Augmentation pipeline (only used if augment=True)
        if self.augment:
            self.transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ])

        # Find all video files (case-insensitive extensions)
        video_paths = []
        raw_labels = []
        for fname in os.listdir(root_dir):
            if fname.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                video_paths.append(os.path.join(root_dir, fname))
                label = self._extract_label_from_filename(fname)
                raw_labels.append(label)

        # Sort to ensure deterministic order (important for multiple instances)
        paired = sorted(zip(video_paths, raw_labels), key=lambda x: x[0])
        self.video_paths, self.raw_labels = zip(*paired) if paired else ([], [])
        self.video_paths = list(self.video_paths)
        self.raw_labels = list(self.raw_labels)

        # Build label to index mapping
        unique_labels = sorted(set(self.raw_labels))
        self.label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
        self.idx_to_label = {i: lbl for lbl, i in self.label_to_idx.items()}
        self.labels = [self.label_to_idx[lbl] for lbl in self.raw_labels]

        print(f"Found {len(self.video_paths)} videos with {len(unique_labels)} unique labels.")

    def _extract_label_from_filename(self, fname):
        """
        Heuristic: remove leading numbers, spaces, and trailing qualifiers like (2), - C, etc.
        Example: "1 hello you how .MOV" -> "hello you how"
                 "2 Home welcome - C.MOV" -> "Home welcome"
                 "3 family big (2).MOV" -> "family big"
        """
        name = os.path.splitext(fname)[0]
        name = re.sub(r'^\d+\s*', '', name)               # remove leading digits and spaces
        name = re.sub(r'\s*\(\d+\)$', '', name)           # remove trailing (2), (3) etc.
        name = re.sub(r'\s*-\s*.*$', '', name)            # remove trailing dash and anything after
        name = name.strip()
        return name

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label_idx = self.labels[idx]

        frames = self._extract_frames(video_path)

        # Convert to tensor (T, C, H, W) and normalize to [0,1]
        video_tensor = torch.from_numpy(frames).float() / 255.0  # (T, H, W, C)
        video_tensor = video_tensor.permute(0, 3, 1, 2)           # (T, C, H, W)

        # Apply data augmentation if enabled (only during training)
        if self.augment:
            augmented_frames = []
            for t in range(video_tensor.size(0)):
                frame = video_tensor[t]          # (C, H, W)
                frame = self.transform(frame)    # apply same transform to each frame
                augmented_frames.append(frame)
            video_tensor = torch.stack(augmented_frames, dim=0)   # (T, C, H, W)

        return video_tensor, label_idx

    def _extract_frames(self, video_path):
        print(f"!!! DEBUG: Now using SEQUENTIAL extraction for {video_path}")
        """
        Extract exactly `self.frames_per_video` frames using uniform sampling
        by reading sequentially. More robust than seeking for problematic videos.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open video {video_path}. Using dummy frames.")
            return np.zeros((self.frames_per_video, self.img_size[1], self.img_size[0], 3), dtype=np.uint8)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print(f"Warning: Video {video_path} has no frames. Using dummy frames.")
            cap.release()
            return np.zeros((self.frames_per_video, self.img_size[1], self.img_size[0], 3), dtype=np.uint8)

        # Calculate which frame indices we want
        target_indices = set(np.linspace(0, total_frames - 1, self.frames_per_video, dtype=int))
        frames = []
        frame_idx = 0

        while frame_idx < total_frames and len(frames) < self.frames_per_video:
            ret, frame = cap.read()
            if not ret:
                # End of video or error – stop reading
                break
            if frame_idx in target_indices:
                try:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, self.img_size, interpolation=cv2.INTER_LINEAR)
                    frames.append(frame)
                except Exception as e:
                    print(f"Warning: Error processing frame {frame_idx} in {video_path}: {e}. Using zeros.")
                    frames.append(np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8))
            frame_idx += 1

        cap.release()

        # If we didn't get enough frames, pad with zeros
        if len(frames) < self.frames_per_video:
            print(f"Warning: Only got {len(frames)} frames from {video_path}, padding with zeros.")
            while len(frames) < self.frames_per_video:
                frames.append(np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8))

        return np.stack(frames, axis=0)


# Quick test (uncomment to run)
if __name__ == '__main__':
    ds = ISLVideoDataset(r'naini/datasets/Indian Sign Language Video and Text dataset', augment=True)
    print(f"Number of samples: {len(ds)}")
    if len(ds) > 0:
        vid, label = ds[0]
        print(f"Video tensor shape: {vid.shape}, label index: {label}, label name: {ds.idx_to_label[label]}")