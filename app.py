import os
import numpy as np
import cv2
from sklearn.neighbors import NearestNeighbors
from flask import Flask, request, render_template
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.decomposition import PCA
import pytesseract
from PIL import Image

# Define the directory where images are stored
IMAGE_DIR = r'C:\Users\LENOVO\Desktop\image-predictor-engine\static\images'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Define the dimensions of the input images
INPUT_SHAPE = (224, 224)

# Load the pre-trained VGG16 model
model = VGG16(weights='imagenet', include_top=False)

# Extract features from all images in the directory
features = []
captions = []
for filename in os.listdir(IMAGE_DIR):
    img_path = os.path.join(IMAGE_DIR, filename)
    img = cv2.imread(img_path)
    img = cv2.resize(img, INPUT_SHAPE)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = np.expand_dims(img_gray, axis=-1)
    img_gray = np.repeat(img_gray, 3, axis=-1)
    img_gray = np.expand_dims(img_gray, axis=0)
    features.append(model.predict(img_gray))

    # Extract text from image using OCR
    img_pil = Image.open(img_path).convert('L')
    text = pytesseract.image_to_string(img_pil)
    captions.append(text)

# Convert features to numpy array
features = np.array(features)

# Reduce the number of dimensions using PCA
pca = PCA(n_components=2)
data_reduced = pca.fit_transform(features.reshape(features.shape[0], -1))

# Apply NearestNeighbors on the reduced data
neigh = NearestNeighbors(n_neighbors=5)
neigh.fit(data_reduced)

# Define a function to find similar images


def find_similar_images(img_path):
    # Load the input image
    img = cv2.imread(img_path)
    img = cv2.resize(img, INPUT_SHAPE)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = np.expand_dims(img_gray, axis=-1)
    img_gray = np.repeat(img_gray, 3, axis=-1)
    img_gray = np.expand_dims(img_gray, axis=0)

    # Extract features from the input image
    query_feature = model.predict(img_gray)[0]
    query_feature_reduced = pca.transform(query_feature.reshape(1, -1))

    # Find the most similar images
    distances, indices = neigh.kneighbors(query_feature_reduced)
    similar_images = [os.path.join(
        '/static/images', os.listdir(IMAGE_DIR)[i]) for i in indices[0]]
    similar_captions = [captions[i] for i in indices[0]]

    return similar_images, similar_captions


# Define the Flask app
app = Flask(__name__, template_folder='templates')

# Define the homepage route


@app.route('/')
def home():
    return render_template('index.html')

# Define the route for uploading an image and finding similar images


@app.route('/find_similar_images', methods=['POST'])
def find_similar_images_route():
    # Get the uploaded image
    uploaded_file = request.files['image']
    uploaded_file.save('tmp.jpg')

    # Find the most similar images
    similar_images, similar_captions = find_similar_images('tmp.jpg')
    print(similar_images, similar_captions)

    # Render the results template with the similar images and captions
    return render_template('results.html', similar_images=similar_images, captions=similar_captions)


if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
