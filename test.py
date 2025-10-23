import os
import librosa
import numpy as np
import joblib

dataset_folder = r"C:\Users\DELL\Desktop\projects\fyp\archive"

model_file = os.path.join(dataset_folder, "rf_urban_sound_model.pkl")
class_mapping_file = os.path.join(dataset_folder, "class_mapping.pkl")

clf = joblib.load(model_file)
class_mapping = joblib.load(class_mapping_file)

test_audio_file = r"C:\Users\DELL\Desktop\projects\fyp\archive\fold3\9223-2-0-2.wav"  
if not os.path.exists(test_audio_file):
    raise FileNotFoundError(f"Test audio not found: {test_audio_file}")

y_audio, sr = librosa.load(test_audio_file, sr=None)

mfccs = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=40)
mfccs_mean = np.mean(mfccs.T, axis=0)
mfccs_mean = mfccs_mean.reshape(1, -1)  

pred_class_id = clf.predict(mfccs_mean)[0]
pred_class_name = class_mapping[pred_class_id]

print(f"Predicted class ID: {pred_class_id}")
print(f"Predicted class name: {pred_class_name}")
