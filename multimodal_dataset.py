import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class MultimodalDataset(Dataset):
    """
    Loads video, text, and optionally audio using soundfile and torchaudio functional.
    """
    def __init__(self, video_root, audio_root=None, frames_per_video=16, img_size=(112, 112),
                 sample_rate=16000, max_audio_len=16000*4):  # 4 seconds
        self.video_root = video_root
        self.audio_root = audio_root
        self.frames_per_video = frames_per_video
        self.img_size = img_size
        self.sample_rate = sample_rate
        self.max_audio_len = max_audio_len
        self.has_audio = audio_root is not None

        # Build audio filename map (cleaned base -> full path)
        self.audio_map = {}
        if self.has_audio:
            for fname in os.listdir(audio_root):
                if fname.lower().endswith('.wav'):
                    base = os.path.splitext(fname)[0].strip().lower()
                    self.audio_map[base] = os.path.join(audio_root, fname)

        self.video_paths = []
        self.raw_labels = []
        self.audio_paths = []

        for fname in os.listdir(video_root):
            if fname.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                video_path = os.path.join(video_root, fname)
                self.video_paths.append(video_path)

                # Extract label from filename
                label = self._extract_label_from_filename(fname)
                self.raw_labels.append(label)

                # Find matching audio
                if self.has_audio:
                    video_base = os.path.splitext(fname)[0].strip().lower()
                    if video_base in self.audio_map:
                        audio_path = self.audio_map[video_base]
                    else:
                        print(f"Warning: No audio match for {fname} (tried '{video_base}.wav')")
                        audio_path = None
                    self.audio_paths.append(audio_path)
                else:
                    self.audio_paths.append(None)

        # Build label mapping
        unique_labels = sorted(set(self.raw_labels))
        self.label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
        self.idx_to_label = {i: lbl for lbl, i in self.label_to_idx.items()}
        self.labels = [self.label_to_idx[lbl] for lbl in self.raw_labels]

        print(f"Found {len(self.video_paths)} multimodal samples. Audio available: {self.has_audio}")
        if self.has_audio:
            matched = sum(1 for p in self.audio_paths if p is not None)
            print(f"Matched {matched} audio files.")

    def _extract_label_from_filename(self, fname):
        """Same heuristic as Phase 1"""
        import re
        name = os.path.splitext(fname)[0]
        name = re.sub(r'^\d+\s*', '', name)
        name = re.sub(r'\s*\(\d+\)$', '', name)
        name = re.sub(r'\s*-\s*.*$', '', name)
        return name.strip()

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        # Video
        video_path = self.video_paths[idx]
        video_frames = self._extract_frames(video_path)          # (T, H, W, C)
        video_tensor = torch.from_numpy(video_frames).float() / 255.0
        video_tensor = video_tensor.permute(0, 3, 1, 2)           # (T, C, H, W)

        # Text
        text_str = self.raw_labels[idx]

        # Audio
        audio_path = self.audio_paths[idx]
        if audio_path is not None:
            try:
                audio_tensor = self._load_audio(audio_path)
            except Exception as e:
                print(f"Error loading audio {audio_path}: {e}. Using zeros.")
                audio_tensor = torch.zeros(1, self.max_audio_len)
        else:
            audio_tensor = torch.zeros(1, self.max_audio_len)

        label_idx = self.labels[idx]

        return {
            'video': video_tensor,
            'text': text_str,
            'audio': audio_tensor,
            'label': label_idx
        }

    def _extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise ValueError(f"Video {video_path} has no frames.")
        indices = np.linspace(0, total_frames-1, self.frames_per_video, dtype=int)
        frames = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                if frames:
                    frame = frames[-1].copy()
                else:
                    frame = np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.img_size, interpolation=cv2.INTER_LINEAR)
            frames.append(frame)
        cap.release()
        return np.stack(frames, axis=0)

    def _load_audio(self, path):
        import soundfile as sf
        # Read with soundfile (returns numpy array and sample rate)
        data, sr = sf.read(path, dtype='float32')  # data shape: (samples,) or (samples, channels)
        # Convert to torch tensor (channels, samples)
        if data.ndim == 1:
            data = data[np.newaxis, :]  # (1, samples)
        else:
            data = data.T  # (channels, samples) because soundfile returns (samples, channels)

        waveform = torch.from_numpy(data)

        # If sample rate differs, resample using torchaudio's functional (no backend required)
        if sr != self.sample_rate:
            import torchaudio.functional as F
            waveform = F.resample(waveform, sr, self.sample_rate)

        # Pad or truncate to fixed length
        if waveform.size(1) > self.max_audio_len:
            waveform = waveform[:, :self.max_audio_len]
        else:
            pad = self.max_audio_len - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        return waveform