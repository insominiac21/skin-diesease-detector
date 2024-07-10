from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load your trained model
model = load_model('path_to_your_trained_model.h5')  # Replace with the actual path to your saved model
label_to_id = {'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3, 'akiec': 4, 'vasc': 5, 'df': 6}  # Replace with your actual labels and IDs
id_to_label = {v: k for k, v in label_to_id.items()}

# Function to preprocess images
def preprocess_image(image):
    try:
        image = Image.open(image).convert('RGB')
        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

# Route to handle image upload and prediction
@app.route('/')
def upload_file():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Convert the file stream to bytes for image processing
        image = preprocess_image(file.stream)
        if image is not None:
            prediction = model.predict(image)
            predicted_label = id_to_label[np.argmax(prediction)]
            return jsonify({'prediction': predicted_label})
        else:
            return jsonify({'error': 'Failed to process image'})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
