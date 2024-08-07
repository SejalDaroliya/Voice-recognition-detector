import os
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Define the dataset path
dataset_path = r'D:/NULLCLASS INTERNSHIP PROJECTS/Voice_gender_detection/dataset'

# Define the features to extract
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccss = mfccs.mean(axis=1)
    return mfccss

# Extract features for female voices
female_path = os.path.join(dataset_path, 'female')
female_wav_path = os.path.join(female_path, 'wav')
female_features = []
female_labels = []

for file in os.listdir(female_wav_path):
    file_path = os.path.join(female_wav_path, file)
    features = extract_features(file_path)
    female_features.append(features)
    female_labels.append(0)  # 0 for female

# Extract features for male voices
male_path = os.path.join(dataset_path, 'male')
male_wav_path = os.path.join(male_path, 'wav')
male_features = []
male_labels = []

for file in os.listdir(male_wav_path):
    file_path = os.path.join(male_wav_path, file)
    features = extract_features(file_path)
    male_features.append(features)
    male_labels.append(1)  # 1 for male

# Combine features and labels
X = np.array(female_features + male_features)
y = np.array(female_labels + male_labels)

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an SVM model
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train)

# Evaluate the model
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Use the model to make predictions
def predict_gender(file_path):
    features = extract_features(file_path)
    features = scaler.transform(features.reshape(1, -1))
    prediction = svm.predict(features)
    if prediction == 0:
        return "Female"
    else:
        return "Male"

# Test the prediction function
file_path = "path/to/test/audio/file.wav"
print("Predicted gender:", predict_gender(file_path))