import os
import re

class AvatarGenerator:
    def __init__(self, avatar_video_folder='avatar_videos/', fingerspelling_folder='fingerspelling/'):
        self.avatar_folder = avatar_video_folder
        self.fingerspelling_folder = fingerspelling_folder
        os.makedirs(avatar_video_folder, exist_ok=True)
        os.makedirs(fingerspelling_folder, exist_ok=True)

        self.label_to_file = {}
        if os.path.exists(avatar_video_folder):
            for fname in os.listdir(avatar_video_folder):
                if fname.lower().endswith('.mp4'):
                    base = os.path.splitext(fname)[0]
                    cleaned = re.sub(r'^\d+\s+', '', base)
                    self.label_to_file[cleaned] = fname

        print(f"AvatarGenerator initialized. Found {len(self.label_to_file)} avatar videos.")
        if self.label_to_file:
            print("Mapping:", self.label_to_file)

    def generate_avatar_video(self, text, output_path=None):
        """
        Return the filename of the avatar video for the given text.
        Tries exact match, then reversed word order, then fallback.
        """
        # Try exact match first
        filename = self.label_to_file.get(text)
        
        # If not found, try reversing the words (handles "big family" vs "family big")
        if filename is None:
            words = text.split()
            if len(words) > 1:
                reversed_text = ' '.join(reversed(words))
                filename = self.label_to_file.get(reversed_text)
                if filename:
                    print(f"Matched reversed order: '{reversed_text}' -> {filename}")
        
        # If still not found, fallback to text + .mp4
        if filename is None:
            filename = f"{text}.mp4"
            print(f"No match for '{text}', using fallback: {filename}")
        
        full_path = os.path.join(self.avatar_folder, filename)
        if not os.path.exists(full_path):
            print(f"Warning: Avatar video file not found at {full_path}")
        return filename

    def play_avatar(self, video_path):
        print("Avatar playback not implemented in web context.")