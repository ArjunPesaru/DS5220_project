import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import gdown
import pickle
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model

# Google Drive file IDs
model_file_id = "https://drive.google.com/file/d/1s05lhAGFIfS7AiXJWmhA5ml_kbsM8an3/view?usp=sharing"
weights_file_id = "https://drive.google.com/file/d/1H7A8lo6DrmYVCHG3zdUsIEtWi8ZtTPIe/view?usp=sharing" 

# Download the model file
def download_file(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)

# Download model and weights
download_file(model_file_id, "my_model.keras")
download_file(weights_file_id, "vgg16_weights_tf_dim_ordering_tf_kernels.h5")

# Load the tokenizer
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Load the trained caption model
caption_model = load_model("my_model.keras")

# Load pre-trained VGG16 with full top layers
modelvgg = VGG16(include_top=True, weights=None)
modelvgg.load_weights("vgg16_weights_tf_dim_ordering_tf_kernels.h5")
modelvgg = Model(inputs=modelvgg.input, outputs=modelvgg.layers[-2].output)

# Set parameters
maxlen = 30  # Ensure this matches your training
start_token = "startseq"
end_token = "endseq"

# Derive index_word if not saved separately
index_word = {i: w for w, i in tokenizer.word_index.items()}

# Extract features function
def extract_features(image):
    image = image.resize((224, 224))
    image_array = img_to_array(image)
    image_array = preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)
    features = modelvgg.predict(image_array, verbose=0)
    return features  # Shape: (1, 4096)

# Generate captions function
def generate_caption(photo):
    in_text = start_token
    for _ in range(maxlen):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=maxlen, padding="post")
        yhat = caption_model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        newword = index_word.get(yhat, None)
        if newword is None:
            break
        in_text += " " + newword
        if newword == end_token:
            break
    return in_text.replace(start_token, "").replace(end_token, "").strip()

# Streamlit UI
st.title("Image Caption Generator")
st.sidebar.title("Options")

# Upload image
uploaded_file = st.file_uploader("Choose an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    features = extract_features(image)
    caption = generate_caption(features)
    st.subheader("Generated Caption:")
    st.write(caption)
