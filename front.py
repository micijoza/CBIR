import streamlit as st
import numpy as np
import cv2
from paths import kth_dir, kth_path
import os
from descriptors import bitdesc
from scipy.spatial import distance
import time


def Calcul(image,n,dis):
    distance_metrics = {
    'euclidean': distance.euclidean,
    'chebyshev': distance.chebyshev,
    'canberra': distance.canberra,
    'cityblock': distance.cityblock
    }
    start_time = time.time()
    signatures = np.load('cbir_signatures_v1.npy')
    distanceList = list()
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    bit_feat = bitdesc(img)
    distance_func = distance_metrics[dis]
    for sign in signatures:
        sign = np.array(sign)[0:-2].astype('float')
        dist = distance_func(bit_feat, sign)
        distanceList.append(dist)
    minDistances = list()
    
    for i in range(n):
        array = np.array(distanceList)
        min_Element_index = np.argmin(array)
        minDistances.append(min_Element_index)
        distanceList[min_Element_index] = np.inf
    end_time = time.time()
    time_passe = end_time - start_time
    st.write(f'About {n} results: ({time_passe:.2f}s)')
    for i in range(0, len(minDistances), 3):
        col1, col2, col3 = st.columns(3)

        for j in range(3):
            if i + j < len(minDistances):
                small = minDistances[i + j]
                img = cv2.imread(signatures[small][-1])
                col = [col1, col2, col3][j]
                col.image(img, channels="BGR", width=200)
    
def load_image_types():
    image_types = {}
    return image_types
def main():
    st.sidebar.header('User Input Parameters')
    value = st.sidebar.number_input('Enter a number :', step=1)
    st.sidebar.write(f"You entered : {value}")
    options = ['euclidean', 'chebyshev', 'canberra', 'cityblock']
    diss = st.sidebar.selectbox('Selected a distance :', options)
    st.sidebar.write(f"You choose : {diss}")
    st.markdown('<p style="color:#5fc4cd;font-size:35px;font-weight:bold; text-decoration:underline;">CBIR : (Content-Based Image Retrieval)</p>', unsafe_allow_html=True)
    st.write(" ")
    upfile = st.file_uploader("Choose an Image file", type=["jpg", "png", "jpeg", "pnm"])
    if upfile is None:
        st.write("No file uploaded")
        st.write("Welcome! Please load an image to start ...")
    else:
        file_bytes = np.asarray(bytearray(upfile.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(image, channels="BGR", caption="Uploaded Query Image", width=200)
        image_types = load_image_types()
        Calcul(image,value,diss)

if __name__ == '__main__':
    main()