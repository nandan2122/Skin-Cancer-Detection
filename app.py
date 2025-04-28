*app.py*

from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import matplotlib.pyplot as plt
import io
import base64
import re

app = Flask(__name__)

# Load the trained model
model_path = 'skin_cancer_model.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model = load_model(model_path)

# Class labels
class_labels = ["benign", "malignant", "no disease"]

# Determine input size from model
input_shape = model.input_shape[1:3]

# Ensure the upload folder exists
UPLOAD_FOLDER = os.path.normpath(os.path.join(os.getcwd(), 'static/uploads'))
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Sanitize filename to avoid invalid characters
def sanitize_filename(filename):
    if not filename or not isinstance(filename, str):
        return "default_upload.jpg"  # Fallback if filename is empty or not a string
    # Keep only alphanumeric characters, underscores, hyphens, and the extension
    sanitized = re.sub(r'[<>:"/\\|?*]+', '_', filename)  # Remove Windows invalid chars
    sanitized = re.sub(r'\s+', '_', sanitized)  # Replace spaces with underscores
    sanitized = re.sub(r'[^\w\-_\.]', '_', sanitized)  # Replace any other invalid chars
    return sanitized.strip('._') or "default_upload.jpg"  # Avoid empty or invalid names


def predict_disease(img_path):
    try:
        print(f"Image path (type: {type(img_path)}): {img_path}")
        if not img_path or not os.path.exists(img_path):
            raise FileNotFoundError(f"Image path is invalid or does not exist: {img_path}")

        img = image.load_img(img_path, target_size=input_shape)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        print(f"Image shape after preprocessing: {img_array.shape}")

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100

        class_probabilities = {class_labels[i]: predictions[0][i] * 100 for i in range(len(class_labels))}

        return class_labels[predicted_class], confidence, class_probabilities

    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None, None


@app.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            print("No file part in the request.")
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print("No selected file.")
            return redirect(request.url)
        if file:
            # Get and sanitize the filename
            original_filename = file.filename if file.filename else "unnamed_file.jpg"
            sanitized_filename = sanitize_filename(original_filename)
            file_path = os.path.normpath(os.path.join(app.config['UPLOAD_FOLDER'], sanitized_filename))

            # Save the file first to ensure it works
            try:
                file.save(file_path)
                if os.path.exists(file_path):
                    print(f"File successfully saved at: {file_path}")
                else:
                    print(f"File save failed: File not found at {file_path}")
                    return "Failed to save the file: File not found after saving."
            except PermissionError as e:
                print(f"Permission denied: {e}")
                return "Failed to save the file: Permission denied."
            except Exception as e:
                print(f"Error saving file: {e}")
                return f"Failed to save the file: {str(e)}"

            # Debugging with safe printing after saving
            try:
                print(f"Original filename: {original_filename.encode('utf-8', errors='replace').decode('utf-8')}")
                print(f"Sanitized filename: {sanitized_filename}")
                print(f"Target file path: {file_path}")
                print(f"Upload folder exists: {os.path.exists(app.config['UPLOAD_FOLDER'])}")
            except Exception as e:
                print(f"Debug printing failed, but file was saved: {e}")

            # Predict the disease
            predicted_class, confidence, class_probabilities = predict_disease(file_path)

            if predicted_class is None:
                return render_template('index.html', prediction="Error processing image.", confidence=0)

            # Generate the bar chart
            plt.figure(figsize=(10, 8))
            plt.bar(class_probabilities.keys(), class_probabilities.values(), color='skyblue')
            plt.ylabel("Confidence (%)")
            plt.xlabel("Disease Classes")
            plt.xticks(rotation=20)
            plt.title("Disease Prediction Confidence")

            # Save the plot to a PNG image in memory
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
            buf.close()

            # Serve the image URL relative to the static folder
            image_url = url_for('static', filename=f'uploads/{sanitized_filename}')

            return render_template('index.html', prediction=predicted_class, confidence=confidence,
                                   plot_url=plot_data, image_url=image_url)

    return render_template('index.html', prediction=None)


if __name__ == '__main__':
    app.run(debug=True)
