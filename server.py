from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import librosa
import subprocess
import csv

app = Flask(__name__)
CORS(app)

# Load model
MODEL_PATH = "yamnet.tflite"
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def convert_to_wav(input_path):
    """Convert WebM/Opus → WAV if needed"""
    output_path = "converted.wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", output_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return output_path
    except Exception as e:
        print("⚠️ FFmpeg conversion failed:", e)
        return input_path


# ✅ Correct way to load YAMNet class labels
def load_labels():
    labels = []
    with open("yamnet_class_map.csv", "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines[1:]:  # skip header
            parts = line.strip().split(",")
            if len(parts) >= 3:
                labels.append(parts[2].strip().strip('"'))
    print(f"Loaded {len(labels)} labels successfully.")

    return labels

labels = load_labels()



def predict_audio(file_path):
    y, sr = librosa.load(file_path, sr=16000, mono=True)
    y = y.astype(np.float32)

    # Match model input length
    expected_len = input_details[0]['shape'][0]
    if len(y) < expected_len:
        y = np.pad(y, (0, expected_len - len(y)))
    else:
        y = y[:expected_len]

    # Add batch dim if needed
    if len(input_details[0]['shape']) > 1:
        y = np.expand_dims(y, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], y)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    # ✅ Get top 3 predictions
    top_indices = np.argsort(output_data)[-3:][::-1]
    top_scores = output_data[top_indices]
    print("Top indices:", top_indices)
    for i in top_indices:
        print(i, "=>", labels[i] if i < len(labels) else "MISSING")


    results = []
    for i, idx in enumerate(top_indices):
        class_name = labels[idx] if idx < len(labels) else "Unknown"
        results.append({
            "rank": i + 1,
            "class_name": class_name,
            "confidence": float(top_scores[i])
        })

    return results


@app.route("/predict", methods=["POST"])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'})
    file = request.files['audio']
    file_path = "temp.wav"
    file.save(file_path)

    predictions = predict_audio(file_path)
    return jsonify({"predictions": predictions})
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    app.run(debug=True)
