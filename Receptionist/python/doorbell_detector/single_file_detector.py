import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa

# Load YAMNet
model = hub.load('https://tfhub.dev/google/yamnet/1')
class_map_path = model.class_map_path().numpy()
class_names = [line.split(',')[2].strip() for line in tf.io.gfile.GFile(class_map_path).readlines()[1:]]

# for i, name in enumerate(class_names):
#     print(f"index: {i}, name:{name}")

# Load your .wav file (resampled to 16kHz)
file_path = '/Users/cademaxwell/Documents/uni/robonotts/healthcare/Receptionist/python/doorbell_detector/data/doorbell/275069__kwahmah_02__doorbell-c.wav'
waveform, sr = librosa.load(file_path, sr=16000)

# 3. Run Inference
# YAMNet can take the whole file at once and it will return scores for every 0.48s
scores, embeddings, spectrogram = model(waveform)

# 4. Analyze results
mean_scores = np.mean(scores, axis=0)
top_class_index = np.argmax(mean_scores)

print(f"--- Analysis for: {file_path} ---")

# could use music and define shape to mimic doorbell if error
print(f"Top sound detected: {class_names[top_class_index]} (Score: {mean_scores[top_class_index]:.2f})")

# (349,Doorbell), (350,Ding-dong), (353,Knock), (477,Ding), (200,Chime), (392,Buzzer)
doorbell_classes = [349,350,353,477,200,392]

for i in doorbell_classes:
    print(f"class: {class_names[i]} score: {mean_scores[i]}")


# Start the stream
# with sd.InputStream(samplerate=FS, channels=1, callback=callback, blocksize=int(FS * DURATION)):
#     print("Press Ctrl+C to stop.")
#     while True: sd.sleep(1000)


