# Enhanced avatar.py with multi-input support
import cv2
import os
import pygame  # for audio playback
from playsound import playsound

class Avatar:
    def __init__(self, avatar_folder='avatars/', audio_folder='audio/'):
        self.avatar_folder = avatar_folder
        self.audio_folder = audio_folder
        # Initialize pygame mixer for smoother audio
        pygame.mixer.init()
    
    def from_text(self, text_label):
        """Generate avatar from predicted text label"""
        print(f"Generating avatar for text: {text_label}")
        self._play_avatar_video(text_label)
        self._play_audio(text_label)
    
    def from_audio(self, audio_path):
        """Generate avatar directly from audio input"""
        # You'd need speech-to-text first
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
        print(f"Transcribed audio: {text}")
        self.from_text(text)
    
    def from_sign_video(self, video_path, model):
        """Generate avatar from sign language video"""
        # Use your Phase 1 model to predict text
        from dataset import ISLVideoDataset
        import torch
        
        # Your existing inference code here
        dataset = ISLVideoDataset('path/to/dataset')
        frames = dataset._extract_frames(video_path)
        video_tensor = torch.from_numpy(frames).float() / 255.0
        video_tensor = video_tensor.permute(0, 3, 1, 2).unsqueeze(0)
        
        with torch.no_grad():
            logits = model(video_tensor)
            pred_idx = torch.argmax(logits, dim=1).item()
            pred_label = dataset.idx_to_label[pred_idx]
        
        self.from_text(pred_label)
    
    def _play_avatar_video(self, text_label):
        """Play pre-recorded avatar video (your existing code)"""
        video_path = os.path.join(self.avatar_folder, f"{text_label}.mp4")
        cap = cv2.VideoCapture(video_path)
        # ... rest of your video playback code
    
    def _play_audio(self, text_label):
        """Play corresponding audio"""
        audio_path = os.path.join(self.audio_folder, f"{text_label}.wav")
        if os.path.exists(audio_path):
            playsound(audio_path)