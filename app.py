import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model dari Google Drive
@st.cache_resource
def load_my_model():
    file_id = "1Pr7CzyLYQOMGWP5_4cpLjokKWQWNGOWr"  # Ganti dengan ID file Drive kamu
    model_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    model_path = get_file("fashion_model.keras", origin=model_url)
    return load_model(model_path)

model = load_my_model()
