import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf

from tempfile import NamedTemporaryFile
from tensorflow.keras.preprocessing import image

st.set_option('deprecation.showfileUploaderEncoding', False)


@st.cache(allow_output_mutation=True)
def loading_model():
    fp = "cnn_pneu_vamp_model.h5"
    model_loader = load_model(fp)
    return model_loader


cnn = loading_model()
st.write("""
# X-Ray Classification [Pneumonia / Normal]
by Shivangi 
""")
