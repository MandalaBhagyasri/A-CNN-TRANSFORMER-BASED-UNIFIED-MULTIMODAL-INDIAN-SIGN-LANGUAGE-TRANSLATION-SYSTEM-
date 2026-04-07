# check_ffmpeg.py
import subprocess
import shutil

def check_ffmpeg():
    """Check if FFmpeg is installed and accessible."""
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        print(f"FFmpeg found at: {ffmpeg_path}")
        # Check version
        result = subprocess.run(['ffmpeg', '-version'], 
                               capture_output=True, text=True)
        print(f"FFmpeg version: {result.stdout.splitlines()[0]}")
        return True
    else:
        print("FFmpeg not found in PATH. Please install FFmpeg:")
        print("1. Download from: https://ffmpeg.org/download.html")
        print("2. Add to system PATH")
        return False

if __name__ == "__main__":
    check_ffmpeg()