import streamlit as st
import numpy as np
from PIL import Image, ImageOps

def predict(image):
    return 0.2

def app():
    # Temporarily using spectogram images
    st.title('Alzheimers Disease')
    file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
    if file is None:
        st.text("Please upload an image file")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        prediction = predict(image)
        num = np.amax(prediction)
        if num >=0.50:
            st.write("Alzheimer's Disease has been diagnosed with: " + str(round(num*100, 3)) + "% Confidence.")
        else:
            st.write("Alzheimer's disease has not been diagnosed with: " + str(round((1.00-num) * 100, 3)) + "% Confidence.")
