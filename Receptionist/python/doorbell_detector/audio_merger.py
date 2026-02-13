import os
from pydub import AudioSegment
from pydub.playback import play

# Define input and output directories
DOORBELL_DIR = "data/doorbell"
BACKGROUND_DIR = "data/background_noise"
OUTPUT_DIR = "data/doorbell_with_background"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Desired output audio length in milliseconds (10 seconds)
OUTPUT_LENGTH_MS = 10 * 1000

def get_audio_files(directory):
    """Helper function to get all audio files from a directory."""
    audio_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith((".wav", ".mp3", ".flac", ".ogg")):  # Add other formats if needed
                audio_files.append(os.path.join(root, file))
    return audio_files

def merge_audio(doorbell_path, background_path, output_path):
    """
    Merges a doorbell sound with background noise.
    The combined audio will be 10 seconds long, with the doorbell in the middle.
    """
    try:
        doorbell = AudioSegment.from_file(doorbell_path)
        background = AudioSegment.from_file(background_path)
    except Exception as e:
        print(f"Error loading audio files: {e}")
        return

    # Ensure background is long enough, loop if necessary
    if len(background) < OUTPUT_LENGTH_MS:
        background = background * (OUTPUT_LENGTH_MS // len(background) + 1)
    
    # Trim background to desired output length
    background = background[:OUTPUT_LENGTH_MS]

    # Calculate start position for doorbell to be in the middle
    doorbell_start_ms = (OUTPUT_LENGTH_MS - len(doorbell)) // 2

    # Overlay doorbell onto the background
    # Using 'average' for overlaying, can adjust 'gain_during_overlay' if needed
    combined_audio = background.overlay(doorbell, position=doorbell_start_ms, gain_during_overlay=-6) # -6dB gain for doorbell

    # Export the combined audio
    try:
        combined_audio.export(output_path, format="wav") # Using WAV for simplicity, can be dynamic
        print(f"Successfully created {output_path}")
    except Exception as e:
        print(f"Error exporting audio to {output_path}: {e}")

def main():
    doorbell_files = get_audio_files(DOORBELL_DIR)
    background_files = get_audio_files(BACKGROUND_DIR)

    if not doorbell_files:
        print(f"No doorbell audio files found in {DOORBELL_DIR}")
        return
    if not background_files:
        print(f"No background audio files found in {BACKGROUND_DIR}")
        return

    for doorbell_file in doorbell_files:
        for background_file in background_files:
            doorbell_name = os.path.splitext(os.path.basename(doorbell_file))[0]
            background_name = os.path.splitext(os.path.basename(background_file))[0]
            
            output_filename = f"{doorbell_name}_with_{background_name}.wav"
            output_path = os.path.join(OUTPUT_DIR, output_filename)

            merge_audio(doorbell_file, background_file, output_path)

if __name__ == "__main__":
    main()
