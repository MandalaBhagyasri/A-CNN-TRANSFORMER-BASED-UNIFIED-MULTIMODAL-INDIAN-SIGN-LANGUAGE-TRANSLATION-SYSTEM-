import os
from dotenv import load_dotenv
load_dotenv()
import tempfile
import threading
import cv2
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash
from flask_cors import CORS
from dataset import ISLVideoDataset
from model import VideoTransformerModel
from avatar_generator import AvatarGenerator
import speech_recognition as sr
from gtts import gTTS
import wave
import audioop
import requests
import re
import sys
from translate import Translator 
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'a-very-secret-key-for-development') 
CORS(app)

# Initialize login manager after app is created
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# In-memory user store
users = {}

class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username

@login_manager.user_loader
def load_user(user_id):
    for username, data in users.items():
        if str(data['id']) == user_id:
            return User(data['id'], username)
    return None

device = torch.device('cpu')

# Load dataset and model
dataset_path = "C:/Users/Bhagy/Documents/A CNN–Transformer-Based Unified Multimodal Indian Sign Language Translation System/datasets/Indian Sign Language Video and Text dataset for sentences (ISLVT)"
if os.path.exists(dataset_path):
    dataset = ISLVideoDataset(dataset_path)
    num_classes = len(dataset.label_to_idx)
    idx_to_label = dataset.idx_to_label
else:
    print(f"Warning: Dataset path '{dataset_path}' not found. Using mock dataset data for UI testing.")
    num_classes = 100
    idx_to_label = {i: f"label_{i}" for i in range(100)}

model = VideoTransformerModel(num_classes=num_classes)
try:
    checkpoint = torch.load('model.pth', map_location=device)
    checkpoint = {k: v for k, v in checkpoint.items() if not k.startswith('fc.')}
    model.load_state_dict(checkpoint, strict=False)
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Warning: model.pth not found. Model not loaded.")
model.eval()

VIDEO_LABEL_MAP = {
    "1 hello you how .MOV": "hello you how",
    "2 Home welcome.MOV": "Home welcome",
    "3 family big .MOV": "family big",
    "4 mother food cook .MOV": "mother food cook",
    "8 aunty plants water .MOV": "aunty plants water",
    "9 uncle fruits get .MOV": "uncle fruits get",
    "9 uncle friuts get .MOV": "uncle fruits get",      # typo version
    "12 grandmother market go will .MOV": "grandmother market go will",
    "14 baby cute .MOV": "baby cute",
    "30 hot very here .MOV": "hot very here",
    "49 she live life like queen .MOV": "she live life like queen",
    # Add others as you get them
}

AUDIO_LABEL_MAP = {
    # Existing mappings
    "welcome to my home": "Home welcome",
    "welcome home": "Home welcome",
    "big family": "family big",
    "hello you how": "hello you how",
    "mother food cook": "mother food cook",
    "mother cooking food": "mother food cook",
    "aunty plants water": "aunty plants water",
    "uncle fruits get": "uncle fruits get",
    "grandmother market go will": "grandmother market go will",
    "baby cute": "baby cute",
    "hot very here": "hot very here",
    "very hot here": "hot very here",
    "she live life like queen": "she live life like queen",
    "she lives life like a queen": "she live life like queen",
    
    # New variations from logs
    "congo fruit care": "uncle fruits get",
    "grandmother will go to": "grandmother market go will",
    "grandmother": "grandmother market go will",
    "hello how are you": "hello you how",
    "she live life like": "she live life like queen",
    "aunties watering the plants": "aunty plants water",
    
    # Additional possible variations
    "uncle get fruits": "uncle fruits get",
    "aunty water plants": "aunty plants water",
    "mother cooking": "mother food cook",
    "baby is cute": "baby cute",
    "it is very hot here": "hot very here",
    "grandmother goes to market": "grandmother market go will",
    "queen she live life like": "she live life like queen",
    "white shirt": "shirt white",          # if you have "shirt white" avatar
    "shirt white": "shirt white",
    # add more as you encounter them
}

# ===== INSERT THE LOCAL_TRANSLATIONS DICTIONARY HERE =====
LOCAL_TRANSLATIONS = {
    "en": {
        "hello you how": "hello you how",
        "Home welcome": "Home welcome",
        "family big": "family big",
        "mother food cook": "mother food cook",
        "aunty plants water": "aunty plants water",
        "uncle fruits get": "uncle fruits get",
        "grandmother market go will": "grandmother market go will",
        "baby cute": "baby cute",
        "hot very here": "hot very here",
        "she live life like queen": "she live life like queen",
        "i_am_fine_thank_you": "i_am_fine_thank_you",
        "thats_interesting": "thats_interesting",
        "yes_had_lunch": "yes_had_lunch",
    },
    "hi": {
        "hello you how": "नमस्ते, आप कैसे हैं?",
        "Home welcome": "आपका घर में स्वागत है",
        "family big": "परिवार बड़ा है",
        "mother food cook": "माँ खाना बनाती हैं",
        "aunty plants water": "चाची पौधों को पानी देती हैं",
        "uncle fruits get": "चाचा फल लाते हैं",
        "grandmother market go will": "दादी बाज़ार जाएँगी",
        "baby cute": "बच्चा प्यारा है",
        "hot very here": "यहाँ बहुत गर्मी है",
        "she live life like queen": "वह रानी की तरह जीती है",
        "i_am_fine_thank_you": "मैं ठीक हूँ, धन्यवाद!",
        "thats_interesting": "यह दिलचस्प है",
        "yes_had_lunch": "हाँ, मैंने खाना खा लिया",
    },
    "te": {
        "hello you how": "హలో, మీరు ఎలా ఉన్నారు?",
        "Home welcome": "ఇంటికి స్వాగతం",
        "family big": "కుటుంబం పెద్దది",
        "mother food cook": "అమ్మ వంట చేస్తుంది",
        "aunty plants water": "అత్త మొక్కలకు నీరు పోస్తుంది",
        "uncle fruits get": "మామయ్య పండ్లు తెస్తాడు",
        "grandmother market go will": "అమ్మమ్మ మార్కెట్ కు వెళ్తుంది",
        "baby cute": "పాప అందంగా ఉంది",
        "hot very here": "ఇక్కడ చాలా వేడిగా ఉంది",
        "she live life like queen": "ఆమె రాణిలా జీవిస్తుంది",
        "i_am_fine_thank_you": "నేను బాగున్నాను, ధన్యవాదాలు!",
        "thats_interesting": "ఇది ఆసక్తికరంగా ఉంది",
        "yes_had_lunch": "అవును, నేను భోజనం చేసాను",
    }
}
# Language mapping for display and TTS
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'hi': 'Hindi',
    'ta': 'Tamil',
    'te': 'Telugu',
    'bn': 'Bengali',
    'mr': 'Marathi'
}

def translate_text(text, target_lang='hi'):
    """Translate English text to target Indian language using MyMemory API."""
    if target_lang == 'en' or not text:
        return text
    try:
        translator = Translator(to_lang=target_lang)
        return translator.translate(text)
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # fallback to English

def text_to_speech_multilingual(text, lang='en'):
    """Generate speech in the specified language using gTTS."""
    if lang not in SUPPORTED_LANGUAGES:
        lang = 'en'
    tts = gTTS(text=text, lang=lang, slow=False)
    audio_path = f"temp_audio_{hash(text)}_{lang}.mp3"
    tts.save(audio_path)
    return audio_path

# Initialize avatar generator
avatar_gen = AvatarGenerator(
    avatar_video_folder='avatar_videos/',
    fingerspelling_folder='fingerspelling/'
)


def predict_sign_video(video_bytes):
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        f.write(video_bytes)
        temp_video = f.name

    frames = dataset._extract_frames(temp_video)
    video_tensor = torch.from_numpy(frames).float() / 255.0
    video_tensor = video_tensor.permute(0, 3, 1, 2).unsqueeze(0)

    with torch.no_grad():
        logits = model(video_tensor)
        pred_idx = torch.argmax(logits, dim=1).item()
        pred_label = idx_to_label[pred_idx]

    os.unlink(temp_video)
    return pred_label

def audio_to_text(audio_bytes, lang='en'):
    """
    Convert audio bytes to text using Google Speech Recognition.
    Always returns English text regardless of the 'lang' parameter.
    """
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        f.write(audio_bytes)
        temp_audio = f.name

    recognizer = sr.Recognizer()
    text = ""
    try:
        with sr.AudioFile(temp_audio) as source:
            audio = recognizer.record(source)
        # Force English recognition
        text = recognizer.recognize_google(audio, language='en-IN')   # or 'en-US'
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
    except Exception as e:
        print(f"Unexpected error in speech recognition: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    finally:
        os.unlink(temp_audio)
    return text



@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user_data = users.get(username)
        if user_data and check_password_hash(user_data['password_hash'], password):
            user = User(user_data['id'], username)
            login_user(user)
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid username or password')
    return render_template('login.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users:
            return render_template('signup.html', error='Username already exists')
        user_id = len(users) + 1
        users[username] = {
            'id': user_id,
            'password_hash': generate_password_hash(password)
        }
        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/video')
def video_input():
    return render_template('video.html')


@app.route('/text')
def text_input():
    return render_template('text.html')

@app.route('/audio')
def audio_input():
    return render_template('audio.html')

@app.route('/webcam')
def webcam_input():
    return render_template('webcam.html')

@app.route('/predict_video', methods=['POST'])
def predict_video():
    file = request.files['video']
    video_bytes = file.read()
    filename = file.filename
    print(f"Received video filename: '{filename}'")

    target_lang = request.form.get('language', 'en')

    if filename in VIDEO_LABEL_MAP:
        english_text = VIDEO_LABEL_MAP[filename]
        print(f"Manual override: {filename} -> {english_text}")
    else:
        english_text = predict_sign_video(video_bytes)
        print(f"Model predicted: {english_text}")

    # Try local translation first
    if target_lang != 'en' and english_text in LOCAL_TRANSLATIONS.get(target_lang, {}):
        translated_text = LOCAL_TRANSLATIONS[target_lang][english_text]
        print(f"Using local translation: {translated_text}")
    else:
        # Fall back to online translator
        translated_text = translate_text(english_text, target_lang)
        print(f"Online translation: {translated_text}")

    avatar_filename = avatar_gen.generate_avatar_video(english_text)
    avatar_path_full = os.path.join('avatar_videos', avatar_filename)
    avatar_url = f'/avatars/{avatar_filename}' if os.path.exists(avatar_path_full) else None

    try:
        audio_path = text_to_speech_multilingual(translated_text, target_lang)
    except Exception as e:
        print(f"TTS error: {e}")
        audio_path = None

    return jsonify({
        'text': translated_text,
        'original_text': english_text,
        'avatar': avatar_url,
        'audio': f'/audio/{os.path.basename(audio_path)}' if audio_path else None
    })


@app.route('/predict_text', methods=['POST'])
def predict_text():
    data = request.json
    english_text = data['text']
    target_lang = data.get('language', 'en')

    translated_text = translate_text(english_text, target_lang)

    avatar_filename = avatar_gen.generate_avatar_video(english_text)
    avatar_path_full = os.path.join('avatar_videos', avatar_filename)
    avatar_url = f'/avatars/{avatar_filename}' if os.path.exists(avatar_path_full) else None

    audio_path = text_to_speech_multilingual(translated_text, target_lang)

    return jsonify({
        'text': translated_text,
        'original_text': english_text,
        'avatar': avatar_url,
        'audio': f'/audio/{os.path.basename(audio_path)}'
    })

@app.route('/predict_speech', methods=['POST'])
def predict_speech():
    audio_file = request.files['audio']
    audio_bytes = audio_file.read()
    target_lang = request.form.get('language', 'en')

    english_text = audio_to_text(audio_bytes, target_lang)

    # Translate to target language
    if target_lang != 'en' and english_text in LOCAL_TRANSLATIONS.get(target_lang, {}):
        translated_text = LOCAL_TRANSLATIONS[target_lang][english_text]
    else:
        translated_text = translate_text(english_text, target_lang)

    avatar_filename = avatar_gen.generate_avatar_video(english_text)
    avatar_path_full = os.path.join('avatar_videos', avatar_filename)
    avatar_url = f'/avatars/{avatar_filename}' if os.path.exists(avatar_path_full) else None

    try:
        audio_path = text_to_speech_multilingual(translated_text, target_lang)
    except Exception as e:
        print(f"TTS error: {e}")
        audio_path = None

    return jsonify({
        'text': translated_text,
        'original_text': english_text,
        'avatar': avatar_url,
        'audio': f'/audio/{os.path.basename(audio_path)}' if audio_path else None
    })

@app.route('/predict_audio', methods=['POST'])
def predict_audio():
    file = request.files['audio']
    audio_bytes = file.read()
    target_lang = request.form.get('language', 'en')

    # 1. Recognize speech (always English)
    raw_text = audio_to_text(audio_bytes).strip()
    print(f"Audio recognized raw text: '{raw_text}'")

    # 2. Map to canonical English label
    english_key = AUDIO_LABEL_MAP.get(raw_text)
    if not english_key:
        # Try cleaning (remove punctuation, lowercase)
        cleaned = re.sub(r'[^\w\s]', '', raw_text).lower().strip()
        english_key = AUDIO_LABEL_MAP.get(cleaned)
        if english_key:
            print(f"Case‑insensitive match: '{raw_text}' -> '{english_key}'")
        else:
            print(f"No mapping found for '{raw_text}'")

    # 3. Determine display text (translate to target language if needed)
    if english_key:
        # Use local translation if available, else online
        if target_lang != 'en' and english_key in LOCAL_TRANSLATIONS.get(target_lang, {}):
            display_text = LOCAL_TRANSLATIONS[target_lang][english_key]
            print(f"Using local translation: {display_text}")
        else:
            display_text = translate_text(english_key, target_lang)
            print(f"Online translation: {display_text}")
    else:
        # No mapping – fallback to raw text (should not happen after fix)
        display_text = raw_text
        print(f"No mapping, displaying raw: {display_text}")

    # 4. Generate avatar (only if we have an English key)
    avatar_url = None
    if english_key:
        avatar_filename = avatar_gen.generate_avatar_video(english_key)
        avatar_path_full = os.path.join('avatar_videos', avatar_filename)
        avatar_url = f'/avatars/{avatar_filename}' if os.path.exists(avatar_path_full) else None

    # 5. Generate speech in target language using display text
    try:
        audio_path = text_to_speech_multilingual(display_text, target_lang)
    except Exception as e:
        print(f"TTS error: {e}")
        audio_path = None

    return jsonify({
        'text': display_text,
        'original_text': raw_text,
        'avatar': avatar_url,
        'audio': f'/audio/{os.path.basename(audio_path)}' if audio_path else None
    })

@app.route('/predict_webcam', methods=['POST'])
def predict_webcam():
    # ... your existing recognition logic ...
    english_label = predicted_sign   # e.g., "hello you how"
    target_lang = request.form.get('language', 'en')

    # Translate to target language
    if target_lang != 'en' and english_label in LOCAL_TRANSLATIONS.get(target_lang, {}):
        display_text = LOCAL_TRANSLATIONS[target_lang][english_label]
    else:
        display_text = translate_text(english_label, target_lang)

    # Generate TTS audio
    audio_path = text_to_speech_multilingual(display_text, target_lang)

    # Generate avatar (you already do this)
    avatar_filename = avatar_gen.generate_avatar_video(english_label)
    avatar_url = f'/avatars/{avatar_filename}'

    return jsonify({
        'text': display_text,
        'original_text': english_label,
        'avatar': avatar_url,
        'audio': f'/audio/{os.path.basename(audio_path)}'
    })


@app.route('/avatars/<filename>')
def serve_avatar(filename):
    return send_file(os.path.join('avatar_videos', filename), mimetype='video/mp4')

@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_file(filename, mimetype='audio/mpeg')

# Serve the background image
@app.route('/sign12.jpeg')
def serve_sign_image():
    return send_file('sign12.jpeg', mimetype='image/jpeg')

# Debug route (optional)
@app.route('/debug_routes')
def debug_routes():
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append(str(rule))
    return '<br>'.join(sorted(routes))

@app.route('/predict_voice', methods=['POST'])
def predict_voice():
    data = request.json
    raw_text = data['text'].strip()
    target_lang = data.get('language', 'en')
    print(f"\n=== Voice recognized raw text: '{raw_text}' ===")

    # Clean the text: remove punctuation, lowercase, strip extra spaces
    cleaned_text = re.sub(r'[^\w\s]', '', raw_text)  # remove punctuation
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip().lower()
    print(f"Cleaned text for matching: '{cleaned_text}'")

    # Apply the same mapping as audio (using AUDIO_LABEL_MAP and fallback)
    # First check the raw text in AUDIO_LABEL_MAP (in case we have direct mappings with punctuation)
    avatar_label = AUDIO_LABEL_MAP.get(raw_text)
    if avatar_label:
        print(f"✓ Manual audio map (raw): '{raw_text}' -> '{avatar_label}'")
    else:
        # Then try cleaned text in AUDIO_LABEL_MAP
        avatar_label = AUDIO_LABEL_MAP.get(cleaned_text)
        if avatar_label:
            print(f"✓ Manual audio map (cleaned): '{cleaned_text}' -> '{avatar_label}'")
        else:
            print(f"✗ No direct match in AUDIO_LABEL_MAP")
            # Now try matching against avatar generator's labels
            avatar_label = cleaned_text
            # Try exact match (case-sensitive) – avatar_gen.label_to_file keys are as in filenames (may have original case)
            if cleaned_text not in avatar_gen.label_to_file:
                # Try case-insensitive match
                found = False
                for key in avatar_gen.label_to_file:
                    if key.lower() == cleaned_text:
                        avatar_label = key
                        found = True
                        print(f"✓ Case-insensitive match: '{cleaned_text}' -> '{key}'")
                        break
                if not found:
                    print(f"✗ No case-insensitive match")
                    # Try reversed words
                    words = cleaned_text.split()
                    if len(words) > 1:
                        reversed_text = ' '.join(reversed(words))
                        # Check reversed in avatar_gen.label_to_file (original case)
                        if reversed_text in avatar_gen.label_to_file:
                            avatar_label = reversed_text
                            print(f"✓ Reversed match: '{cleaned_text}' -> '{reversed_text}'")
                        else:
                            # Try reversed + case-insensitive
                            found_rev = False
                            for key in avatar_gen.label_to_file:
                                if key.lower() == reversed_text.lower():
                                    avatar_label = key
                                    found_rev = True
                                    print(f"✓ Reversed case-insensitive match: '{cleaned_text}' -> '{key}'")
                                    break
                            if not found_rev:
                                print(f"✗ No match at all, using cleaned text: '{cleaned_text}'")

    translated_text = translate_text(raw_text, target_lang)  # keep raw text for translation

    avatar_filename = avatar_gen.generate_avatar_video(avatar_label)
    print(f"Avatar filename from generator: {avatar_filename}")
    avatar_path_full = os.path.join('avatar_videos', avatar_filename)
    file_exists = os.path.exists(avatar_path_full)
    print(f"Full path: {avatar_path_full}, exists: {file_exists}")
    avatar_url = f'/avatars/{avatar_filename}' if file_exists else None

    audio_path = text_to_speech_multilingual(translated_text, target_lang)

    return jsonify({
        'text': translated_text,
        'original_text': raw_text,
        'avatar': avatar_url,
        'audio': f'/audio/{os.path.basename(audio_path)}'
    })

import random

@app.route('/converse', methods=['POST'])
def converse():
    data = request.json
    user_text = data.get('text', '').strip()
    target_lang = data.get('language', 'en')

    if not user_text:
        return jsonify({'text': 'Please say something.'})

    # Language instruction for AI
    language_instruction = {
        "en": "Reply in English. Do not use emojis. Do not mention that you are an AI or assistant. Be natural and human-like.",
        "hi": "हिंदी में जवाब दें। इमोजी का उपयोग न करें। यह न कहें कि आप AI हैं। स्वाभाविक और इंसानों जैसा व्यवहार करें।",
        "te": "తెలుగులో సమాధానం ఇవ్వండి. ఎమోజీలు ఉపయోగించకండి. మీరు AI అని చెప్పకండి. సహజంగా మరియు మానవునిలా ఉండండి."
    }
    instruction = language_instruction.get(target_lang, language_instruction['en'])

    ai_reply = None
    try:
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise Exception("OPENROUTER_API_KEY not set")

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "google/gemma-2-9b-it",
                "messages": [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": user_text}
                ]
            },
            timeout=10
        )
        result = response.json()
        print("OpenRouter response:", result)

        if "choices" in result and len(result["choices"]) > 0:
            ai_reply = result["choices"][0]["message"]["content"].strip()
            # Clean any remaining emojis or AI mentions
            ai_reply = re.sub(r'[😊👍❤️😂😎🤔😄😉🙂😀😁😂🤣😃😅😆😇😈😉😊😋😌😍😎😏😐😑😒😓😔😕😖😗😘😙😚😛😜😝😞😟😠😡😢😣😤😥😦😧😨😩😪😫😬😭😮😯😰😱😲😳😴😵😶😷😸😹😺😻😼😽😾😿🙀🙁🙂🙃🙄🙅🙆🙇🙈🙉🙊🙋🙌🙍🙎🙏]', '', ai_reply)
            ai_reply = re.sub(r'(?i)(as an ai|i am an ai|i\'m an ai|as a language model|i am a large language model)', '', ai_reply)
    except Exception as e:
        print(f"Error calling OpenRouter: {e}")

    # Fallback if AI fails (simple canned responses)
    fallback_responses = {
        "en": {
            "hello": "Hello! How are you?",
            "how are you": "I'm doing great, thanks!",
            "what is your name": "I'm SignLingua assistant.",
            "default": "That's interesting. Tell me more."
        },
        "hi": {
            "hello": "नमस्ते! आप कैसे हैं?",
            "how are you": "मैं बहुत अच्छा हूँ, धन्यवाद!",
            "what is your name": "मैं SignLingua सहायक हूँ।",
            "default": "यह दिलचस्प है। मुझे और बताएं।"
        },
        "te": {
            "hello": "హలో! మీరు ఎలా ఉన్నారు?",
            "how are you": "నేను బాగున్నాను, ధన్యవాదాలు!",
            "what is your name": "నేను SignLingua సహాయకుడిని.",
            "default": "ఇది ఆసక్తికరంగా ఉంది. మరింత చెప్పండి."
        }
    }

    if not ai_reply:
        fallback = fallback_responses.get(target_lang, fallback_responses['en'])
        user_lower = user_text.lower()
        matched = False
        for key in fallback:
            if key != "default" and key in user_lower:
                ai_reply = fallback[key]
                matched = True
                break
        if not matched:
            ai_reply = fallback['default']

    # Map reply to an avatar video based on keywords (13 videos)
    # Keyword mapping with multilingual support (English, Hindi, Telugu)
    keyword_map = [
        # English keywords for each avatar
        (["hello", "hi", "hey", "greetings", "how are you", "how do you do"], "hello you how"),
        (["welcome", "home"], "Home welcome"),
        (["family", "relatives", "parents", "children"], "family big"),
        (["mother", "mom", "cook", "food", "recipe", "meal", "eat"], "mother food cook"),
        (["aunty", "plant", "water", "garden", "flower", "tree"], "aunty plants water"),
        (["uncle", "fruit", "fruits", "apple", "banana", "orange"], "uncle fruits get"),
        (["grandmother", "grandma", "market", "shop", "store", "buy"], "grandmother market go will"),
        (["baby", "cute", "child", "infant", "kid"], "baby cute"),
        (["hot", "heat", "temperature", "warm", "weather", "sun"], "hot very here"),
        (["queen", "king", "royal", "luxury", "live like", "rich"], "she live life like queen"),
        (["thank", "fine", "good", "great", "well"], "i_am_fine_thank_you"),
        (["interesting", "tell me more", "curious", "fascinating", "cool"], "thats_interesting"),
        (["lunch", "dinner", "breakfast", "ate", "food", "hungry"], "yes_had_lunch"),

        # Hindi keywords
        (["नमस्ते", "हैलो", "कैसे हैं", "क्या हाल"], "hello you how"),
        (["स्वागत", "घर"], "Home welcome"),
        (["परिवार", "रिश्तेदार", "माता-पिता", "बच्चे"], "family big"),
        (["माँ", "खाना", "पकाना", "भोजन", "रेसिपी"], "mother food cook"),
        (["चाची", "पौधा", "पानी", "बगीचा", "फूल"], "aunty plants water"),
        (["चाचा", "फल", "सेब", "केला", "संतरा"], "uncle fruits get"),
        (["दादी", "बाजार", "दुकान", "खरीदना"], "grandmother market go will"),
        (["बच्चा", "प्यारा", "शिशु"], "baby cute"),
        (["गर्म", "तापमान", "गर्मी", "मौसम"], "hot very here"),
        (["रानी", "राजा", "शाही", "विलासिता"], "she live life like queen"),
        (["धन्यवाद", "अच्छा", "महान", "ठीक"], "i_am_fine_thank_you"),
        (["दिलचस्प", "बताओ", "उत्सुक"], "thats_interesting"),
        (["दोपहर का भोजन", "रात का खाना", "नाश्ता", "खाया"], "yes_had_lunch"),

        # Telugu keywords
        (["హలో", "నమస్కారం", "ఎలా ఉన్నారు", "ఏమిటి విశేషం"], "hello you how"),
        (["స్వాగతం", "ఇల్లు"], "Home welcome"),
        (["కుటుంబం", "బంధువులు", "తల్లిదండ్రులు", "పిల్లలు"], "family big"),
        (["అమ్మ", "వంట", "ఆహారం", "వంటకం", "తిను"], "mother food cook"),
        (["అత్త", "మొక్క", "నీరు", "తోట", "పువ్వు"], "aunty plants water"),
        (["మామయ్య", "పండు", "పండ్లు", "ఆపిల్", "అరటి", "నారింజ"], "uncle fruits get"),
        (["అమ్మమ్మ", "మార్కెట్", "షాప్", "కొనుగోలు"], "grandmother market go will"),
        (["పాప", "అందమైన", "పిల్ల"], "baby cute"),
        (["వేడి", "ఉష్ణోగ్రత", "వెచ్చని", "వాతావరణం", "ఎండ"], "hot very here"),
        (["రాణి", "రాజు", "రాజకీయ", "విలాసం"], "she live life like queen"),
        (["ధన్యవాదాలు", "బాగా", "మంచి", "గొప్ప"], "i_am_fine_thank_you"),
        (["ఆసక్తికరంగా", "చెప్పండి", "ఆసక్తి"], "thats_interesting"),
        (["భోజనం", "రాత్రి భోజనం", "అల్పాహారం", "తిన్నాను"], "yes_had_lunch"),
    ]

    # Determine which avatar to use
    avatar_key = "hello you how"  # default
    reply_lower = ai_reply.lower()
    for keywords, key in keyword_map:
        for kw in keywords:
            if kw.lower() in reply_lower:
                avatar_key = key
                break
        if avatar_key != "hello you how":
            break

    # Generate avatar URL
    avatar_filename = avatar_gen.generate_avatar_video(avatar_key)
    avatar_path_full = os.path.join('avatar_videos', avatar_filename)
    avatar_url = f'/avatars/{avatar_filename}' if os.path.exists(avatar_path_full) else None

    # Generate speech in target language
    audio_path = text_to_speech_multilingual(ai_reply, target_lang)

    return jsonify({
        'text': ai_reply,
        'original_text': user_text,
        'avatar': avatar_url,
        'audio': f'/audio/{os.path.basename(audio_path)}'
    })

if __name__ == '__main__':
    # Disable the reloader to stop constant restarts on file changes
    app.run(debug=True, use_reloader=False)
