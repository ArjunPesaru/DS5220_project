# DS5220_project
# TITLE: Machine Learning Techniques to predict Captions
# Working Model: https://image-caption9-gen.streamlit.app/
Author: Arjun Pesaru & Hiranmai Devarasetty



This project demonstrates an image caption generator using a **VGG16 encoder** and **LSTM-based decoder**. A Streamlit app is provided for a user-friendly interface to upload images and generate captions.

---

## **Setup Instructions**

### Step 1: Download the Flickr8k Dataset
1. Download the **Flickr8k dataset** from Kaggle:  
   [https://www.kaggle.com/datasets/jaykumar2862/flicker-8k](https://www.kaggle.com/datasets/jaykumar2862/flicker-8k)
2. Place the downloaded dataset in the project directory under a folder named `Flickr_Data`.

---

### Step 2: Install Required Libraries
Ensure you have Python installed (preferably version 3.8 or above). Then, install the required Python libraries by running:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file includes essential libraries like:
- TensorFlow
- Streamlit
- gdown

---

### Step 3: Download Pre-trained Model and Weights
The app requires two pre-trained files stored in Google Drive:
1. **Model file**: `my_model.keras`
2. **VGG16 weights**: `vgg16_weights_tf_dim_ordering_tf_kernels.h5`

These files will automatically download when you run the app. Ensure you have a stable internet connection.

---

### Step 4: Run the Streamlit App
To launch the app, execute the following command in your terminal or command prompt:

```bash
streamlit run app.py
```

---

## **Streamlit App Workflow**

1. **Uploading an Image**:
   - Use the app's file uploader to upload an image in `.jpg`, `.jpeg`, or `.png` format.

2. **Caption Generation**:
   - The app extracts features from the uploaded image using **VGG16**.
   - The pre-trained model generates a caption using the features and tokenizer.

3. **Output**:
   - The generated caption is displayed below the uploaded image.

---

### **Check Our Working Site**
If you want to test the application directly, visit our deployed site:  
[https://image-caption9-gen.streamlit.app/](https://image-caption9-gen.streamlit.app/)

---

## **File Descriptions**

| File Name                                | Description                                                                 |
|-----------------------------------------|-----------------------------------------------------------------------------|
| `app.py`                                | Main file to run the Streamlit app.                                         |
| `my_model.keras`                        | Pre-trained image captioning model.(avaliable on my drive:)                                      |
| `vgg16_weights_tf_dim_ordering_tf_kernels.h5` | Pre-trained VGG16 weights for feature extraction.(avaliable on my drive)                          |
| `tokenizer.pkl`                         | Tokenizer file containing mappings for vocabulary and captions.            |
| `requirements.txt`                      | List of Python libraries required to run the app.                          |
| `Flickr_Data/`                          | Folder containing the Flickr8k dataset (to be downloaded separately from kaggle).      |

---

## **Notes**
- Ensure the `Flickr_Data` folder, `my_model.keras`, and `vgg16_weights_tf_dim_ordering_tf_kernels.h5` are available in the working directory.
- The app will automatically download the model and weights from Google Drive if not present locally.

For any issues, ensure all dependencies are installed and files are correctly placed in the directory.

---

Enjoy generating captions! 😊


