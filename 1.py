import streamlit as st
import os
import base64
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from skimage.feature import hog
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn import tree, svm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tkinter.filedialog import askopenfilename

# Streamlit title
st.title("Blood Type Classification")
def add_bg_from_local(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    bg_image = f"data:predit/jpg;base64,{encoded_string}"
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{bg_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function with your image path
add_bg_from_local("C:/Users/Malav/OneDrive/Desktop/Project/5. Blood group detection using fingerprint/5. Blood group detection using fingerprint/Blood Finger/image/predit.jpg")  # Make sure to provide the correct image filename



# File Upload in Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg" , "BMP"])

if uploaded_file is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Display Image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Resize and Convert to Grayscale
    img_resize2 = cv2.resize(img, (50, 50))
    if len(img_resize2.shape) == 3:
        img_resize2 = cv2.cvtColor(img_resize2, cv2.COLOR_BGR2GRAY)

    # Load dataset
    dataset_path = "Dataset"
    blood_types = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]
    dot = []
    labels = []

    for i, blood_type in enumerate(blood_types):
        path = os.path.join(dataset_path, blood_type)
        if os.path.exists(path):
            for img_name in os.listdir(path):
                try:
                    img_path = os.path.join(path, img_name)
                    img_data = plt.imread(img_path)
                    img_resized = cv2.resize(img_data, (50, 50))
                    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                    dot.append(np.array(img_resized))
                    labels.append(i)
                except:
                    pass

    # Extract Features
    Trainfea = [[np.mean(img), np.std(img), np.var(img)] for img in dot]
    Testfea = [np.mean(img_resize2), np.std(img_resize2), np.var(img_resize2)]

    # Train Decision Tree Classifier
    clf = tree.DecisionTreeClassifier()
    clf.fit(Trainfea, labels)
    Class_val_d = clf.predict(Trainfea)

    # Model Accuracy
    ACC_ml = accuracy_score(labels, Class_val_d)
    st.write(f"**Accuracy of ML Model:** {ACC_ml * 100:.2f}%")

    # Confusion Matrix
    conf_matrix = confusion_matrix(labels, Class_val_d)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va="center", ha="center", size="xx-large")

    plt.xlabel("Predictions", fontsize=14)
    plt.ylabel("Actuals", fontsize=14)
    plt.title("Confusion Matrix", fontsize=14)
    st.pyplot(fig)

    # Classification Report
    report = classification_report(labels, Class_val_d, target_names=blood_types)
    st.text(report)

    # Predict Uploaded Image
    predicted_label = clf.predict([Testfea])[0]
    st.write(f"**Predicted Blood Type:** {blood_types[predicted_label]}")
