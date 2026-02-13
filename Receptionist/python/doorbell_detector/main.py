import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import os

# Load YAMNet
model = hub.load('https://tfhub.dev/google/yamnet/1')
class_map_path = model.class_map_path().numpy()
class_names = [line.split(',')[2].strip() for line in tf.io.gfile.GFile(class_map_path).readlines()[1:]]

# Define target directory, doorbell classes, and confidence threshold
AUDIO_DIR = "data/doorbell_with_background"
# (349,Doorbell), (350,Ding-dong), (353,Knock), (477,Ding), (200,Chime), (392,Buzzer)
DOORBELL_CLASSES = [349, 350, 353, 477, 200, 392]
CONFIDENCE_THRESHOLD = 0.3

def get_audio_files(directory):
    """Helper function to get all audio files from a directory."""
    audio_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith((".wav", ".mp3", ".flac", ".ogg")):  # Add other formats if needed
                audio_files.append(os.path.join(root, file))
    return audio_files

def analyze_audio_for_doorbell(file_path):
    """
    Analyzes an audio file for the presence of doorbell-related sounds
    above a certain confidence threshold.
    """
    try:
        waveform, sr = librosa.load(file_path, sr=16000)
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return False # Assume no doorbell if file can't be loaded

    # YAMNet can take the whole file at once and it will return scores for every 0.48s
    scores, embeddings, spectrogram = model(waveform)

    doorbell_detected = False
    for score_per_frame in scores:
        for class_index in DOORBELL_CLASSES:
            if score_per_frame[class_index] > CONFIDENCE_THRESHOLD:
                doorbell_detected = True
                break # Found a doorbell class above threshold in this frame
        if doorbell_detected:
            break # Found a doorbell class above threshold in any frame

    return doorbell_detected

def main():
    audio_files_to_analyze = get_audio_files(AUDIO_DIR)

    if not audio_files_to_analyze:
        print(f"No audio files found in {AUDIO_DIR} to analyze.")
        return

    print(f"Analyzing audio files in {AUDIO_DIR} for doorbell sounds...")
    for audio_file_path in audio_files_to_analyze:
        filename = os.path.basename(audio_file_path)
        is_doorbell_present = analyze_audio_for_doorbell(audio_file_path)
        print(f"{filename}: {is_doorbell_present}")

if __name__ == "__main__":
    main()
