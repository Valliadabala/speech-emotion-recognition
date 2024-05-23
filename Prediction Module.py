import os
import numpy as np
import librosa
import tensorflow as tf
from keras.models import model_from_json
import sounddevice as sd
import json
import tkinter as tk

save_dir = "C:/Users/valli adabala/OneDrive/Desktop/pro files/"

with open(os.path.join(save_dir, 'model_architecture (1).json'), 'r') as f:
    config = json.load(f)

for layer in config['config']['layers']:
    if 'batch_shape' in layer['config']:
        del layer['config']['batch_shape']

with open(os.path.join(save_dir, 'model_architecture (1).json'), 'w') as f:
    json.dump(config, f)

with open(os.path.join(save_dir, "model_architecture (1).json"), "r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights(os.path.join(save_dir, "model_weights.h5"))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

emotion_labels = {0: 'neutral', 1: 'happy', 2: 'sad', 3: 'fear', 4: 'angry', 5: 'disgust', 6: 'surprise'}

def extract_features(data, sample_rate):
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))
    return result

def get_features(data, sample_rate):
    res1 = extract_features(data, sample_rate)
    result = np.array(res1)
    return result

def predict_emotion(audio_data, sample_rate=22050):
    features = get_features(audio_data, sample_rate)
    features = np.expand_dims(features, axis=0)
    features = np.expand_dims(features, axis=2)
    prediction = model.predict(features)
    predicted_emotion_index = np.argmax(prediction)
    predicted_emotion = emotion_labels[predicted_emotion_index]
    return predicted_emotion

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    global audio_data
    audio_data = indata[:, 0]

def start_recording():
    global stream
    stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=22050)
    stream.start()
    start_button.config(state=tk.DISABLED)
    stop_button.config(state=tk.NORMAL)
    predict_button.config(state=tk.DISABLED)

def stop_recording():
    stream.stop()
    stream.close()
    start_button.config(state=tk.NORMAL)
    stop_button.config(state=tk.DISABLED)
    predict_button.config(state=tk.DISABLED)
    predict()

def predict():
    emotion = predict_emotion(audio_data)
    emotion_label.config(text=f"Predicted Emotion: {emotion}", fg='blue')

root = tk.Tk()
root.title("Emotion Prediction by valli's")


root.config(bg='white')
font_style = ("Lucida Bright", 16)

button_frame = tk.Frame(root, bg='white')
button_frame.pack(pady=10)

start_button = tk.Button(button_frame, text="Start Recording", command=start_recording, state=tk.NORMAL, bg='black', fg='white', font=font_style)
start_button.pack( pady=20)

stop_button = tk.Button(button_frame, text="Stop Recording", command=stop_recording, state=tk.DISABLED, bg='black', fg='white', font=font_style)
stop_button.pack(pady=20)

predict_button = tk.Button(button_frame, text="Predict Emotion", command=predict, state=tk.DISABLED, bg='black', fg='white', font=font_style)
predict_button.pack(pady=20)

emotion_label = tk.Label(root, text="Predicted Emotion: None", bg='lightgray', font=font_style)
emotion_label.pack(pady=20)

root.mainloop()
