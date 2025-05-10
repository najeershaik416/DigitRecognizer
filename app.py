from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import base64

# Load the trained CNN model
model = tf.keras.models.load_model('mnist_cnn_model.h5')

# Initialize Flask app
app = Flask(__name__)

# Function to process image and make prediction
def predict_digit(image):
    image = image.convert("L").resize((28, 28))  # Convert to grayscale and resize
    img_array = np.array(image, dtype=np.float32).reshape(1, 28, 28, 1)  # Reshape for the model
    img_array /= 255.0  # Normalize the image
    prediction = model.predict(img_array)  # Get the model prediction
    return np.argmax(prediction)  # Return the index with the highest probability

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle the uploaded image
@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image file
    file = request.files['image']
    
    # Convert the image to grayscale and resize it to 28x28 (MNIST size)
    img = Image.open(file.stream)
    
    # Predict the digit
    prediction = predict_digit(img)
    
    # Convert image to base64 for display in HTML
    img_base64 = image_to_base64(img)
    
    return render_template('predict.html', prediction=prediction, img_base64=img_base64)

# Function to convert image to base64 for display in HTML
def image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

if __name__ == "__main__":
    app.run(debug=True)
