import os
import pandas as pd
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

dataset_folder = r"C:\Users\DELL\Desktop\projects\fyp\archive"
metadata_file = os.path.join(dataset_folder, "UrbanSound8K.csv")

metadata = pd.read_csv(metadata_file)

X = []
y = []

print("Extracting features from WAV files...")

for _, row in metadata.iterrows():
    fold = row['fold']
    file_name = row['slice_file_name']
    class_id = row['classID']
    
    file_path = os.path.join(dataset_folder, f"fold{fold}", file_name)
    
    if os.path.exists(file_path):
        y_audio, sr = librosa.load(file_path, sr=None)
        
        mfccs = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=40)  
        mfccs_mean = np.mean(mfccs.T, axis=0) 
        X.append(mfccs_mean)
        y.append(class_id)
    else:
        print(f"File not found: {file_path}")

X = np.array(X)
y = np.array(y)

print(f"Features shape: {X.shape}, Labels shape: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

import joblib

model_file = os.path.join(dataset_folder, "rf_urban_sound_model.pkl")
joblib.dump(clf, model_file)
print(f"Model saved to {model_file}")

class_mapping = dict(zip(metadata['classID'], metadata['class']))
joblib.dump(class_mapping, os.path.join(dataset_folder, "class_mapping.pkl"))
