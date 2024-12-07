import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
from tensorflow.keras.preprocessing.image import img_to_array

# Load the tokenizer
with open("/Users/arjunpesaru/Desktop/sml_p/tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Load the trained caption model
caption_model = load_model("/Users/arjunpesaru/Desktop/sml_p/my_model.keras")  # Replace with "caption_model.keras" if saved in .keras format

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

# Load pre-trained VGG16 with full top layers
modelvgg = VGG16(include_top=True, weights=None)

# Load locally saved weights
modelvgg.load_weights("/Users/arjunpesaru/Desktop/sml_p/vgg16_weights_tf_dim_ordering_tf_kernels.h5")

# Remove the final classification layer to extract features
modelvgg = Model(inputs=modelvgg.input, outputs=modelvgg.layers[-2].output)  # -2 corresponds to the second last layer

# Verify the output shape
print(modelvgg.output_shape)  # Should be (None, 4096)






# Set parameters (ensure these match what you used during training)
maxlen = 29  # Replace with your training max sequence length
start_token = "startseq"
end_token = "endseq"

# Derive index_word if not saved separately
index_word = {i: w for w, i in tokenizer.word_index.items()}

def extract_features(image):
    """
    Extract features from the image using the modified VGG16 model.
    """
    image = image.resize((224, 224))  # Resize to VGG16 input size
    image_array = img_to_array(image)  # Convert to numpy array
    image_array = preprocess_input(image_array)  # Preprocess the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    features = modelvgg.predict(image_array, verbose=0)  # Extract features
    return features  # Should have shape (1, 4096)




def generate_caption(photo):
    """
    Generate a caption using greedy search.
    photo: Extracted features from the image (shape: (1, 4096)).
    """
    # Adjust maxlen to match model's expected input
    maxlen = 30  # Changed from 29 to 30

    in_text = start_token  # Initialize with 'startseq'
    for _ in range(maxlen):  # Iterate for maxlen steps
        # Convert the current text into a sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        
        # Pad the sequence to ensure consistent input length
        # Explicitly use maxlen=30 to match model's expectation
        sequence = pad_sequences([sequence], maxlen=maxlen, padding='post')
        
        # Predict the next word
        yhat = caption_model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)  # Get the index of the most probable word
        newword = index_word.get(yhat, None)  # Map the index to a word
        
        if newword is None:
            break
        
        in_text += " " + newword
        
        if newword == end_token:  # Stop if 'endseq' is predicted
            break
    
    # Remove the start and end tokens from the generated caption
    final_caption = in_text.replace(start_token, '').replace(end_token, '').strip()
    return final_caption


# Streamlit UI
st.title("Image Caption Generator")
st.sidebar.title("Options")

# Upload image
uploaded_file = st.file_uploader("Choose an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Extract features using VGG16
    features = extract_features(image)
    
    # Generate caption
    caption = generate_caption(features)

    # Display the generated caption
    st.subheader("Generated Caption:")
    st.write(caption)
