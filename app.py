import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model
model = load_model("fashion_model.keras")

# Daftar label kelas (ubah sesuai dataset-mu)
labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
          "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Judul
st.title("👗👟 Fashion Classification App")
st.write("Upload gambar fashion (28x28 pixel, grayscale) untuk diprediksi:")

# Upload gambar
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar
    img = Image.open(uploaded_file).convert('L')  # Convert ke grayscale
    st.image(img, caption='Gambar yang diupload', use_column_width=True)

    # Preprocessing sesuai model
    img = img.resize((28, 28))
    img_array = image.img_to_array(img)
    img_array = img_array.reshape((1, 28, 28, 1)) / 255.0

    # Prediksi
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.success(f"Hasil Prediksi: **{labels[class_idx]}**")
    st.write(f"Confidence: {confidence:.2f}%")
