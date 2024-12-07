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
model_file_id = "1s05lhAGFIfS7AiXJWmhA5ml_kbsM8an3"
weights_file_id = "1H7A8lo6DrmYVCHG3zdUsIEtWi8ZtTPIe"

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
st.set_page_config(
    page_title="Image Caption Generator",
    page_icon=":camera:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Header section
st.markdown(
    """
    <style>
        .main-header {
            font-size: 40px;
            color: #4CAF50;
            text-align: center;
        }
        .sub-header {
            font-size: 20px;
            color: #555;
            text-align: center;
        }
        .footer {
            text-align: center;
            color: #888;
            margin-top: 50px;
        }
    </style>
    <div class="main-header">📸 Image Caption Generator</div>
    <div class="sub-header">Upload an image and let AI describe it!</div>
    """,
    unsafe_allow_html=True,
)

# Sidebar section
st.sidebar.markdown(
    """
    ## Instructions
    - Upload an image using the button below.
    - Wait for the AI to process the image.
    - View the generated caption on the main screen.
    """
)

# Upload image
uploaded_file = st.file_uploader(
    "Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    col1, col2 = st.columns(2)
    with col1:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        # Extract features and generate caption
        features = extract_features(image)
        caption = generate_caption(features)

        # Display the generated caption
        st.subheader("Generated Caption:")
        st.success(caption)
# Footer section
st.markdown(
    """
    <div class="footer">
        Built by <b>Arjun Pesaru</b> and <b>Hiranmai Devarasetty</b>
    </div>
    """,
    unsafe_allow_html=True,
)
