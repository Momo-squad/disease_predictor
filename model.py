import numpy as np 
import cv2
import pickle
import matplotlib.pyplot as plt
import os
from numpy import asarray
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
# Just for commit
os.chdir(os.path.dirname(__file__))

model = pickle.load(open("/run/media/bb/NotNSFW/Code/still_learning_ML/disease_predictor/venv/dis_classify.pkl", 'rb'))

dis_arr = ["Apple___Apple_scab","Apple___Black_rot",
 "Apple___Cedar_apple_rust",
 "Apple___healthy",
 "Blueberry___healthy",
'Cherry_(including_sour)___Powdery_mildew',
'Cherry_(including_sour)___healthy',
'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
'Corn_(maize)___Common_rust_',
'Corn_(maize)___Northern_Leaf_Blight',
'Corn_(maize)___healthy',
 "Grape___Black_rot",
'Grape___Esca_(Black_Measles)',
'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 "Grape___healthy",
'Orange___Haunglongbing_(Citrus_greening)',
 "Peach___Bacterial_spot",
 "Peach___healthy",
 "Pepper_bell___Bacterial_spot",
 "Pepper,_bell___healthy",
 "Potato___Early_blight",
 "Potato___Late_blight",
 "Potato___healthy",
 "Raspberry___healthy",
 "Soybean___healthy",
 "Squash___Powdery_mildew",
 "Strawberry___Leaf_scorch",
 "Strawberry___healthy",
 "Tomato___Bacterial_spot",
 "Tomato___Early_blight",
 "Tomato___Late_blight",
 "Tomato___Leaf_Mold",
 "Tomato___Septoria_leaf_spot",
'Tomato___Spider_mites Two-spotted_spider_mite',
 "Tomato___Target_Spot",
 "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
 "Tomato___Tomato_mosaic_virus",
 "Tomato___healthy"
] 
DEFAULT_IMAGE_SIZE = tuple((256, 256))



def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None:
            image = cv2.resize(image, DEFAULT_IMAGE_SIZE)   
            return img_to_array(image)
        else:
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None



def predict_disease(image_path):
    image_array = convert_image_to_array(image_path)
    np_image = np.array(image_array, dtype=np.float16) / 225.0
    np_image = np.expand_dims(np_image,0)
    plt.imshow(plt.imread(image_path))
    
    resutlt = np.argmax(model.predict(np_image))
    print(dis_arr[resutlt])
    # print([i for i,prob in enumerate(resutlt) if prob > 0.5])
    # result = model.predict_classes(np_image)
    print(resutlt)


predict_disease("/run/media/bb/NotNSFW/Code/dataset/newplant/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/Apple___Cedar_apple_rust/0a41c25a-f9a6-4c34-8e5c-7f89a6ac4c40___FREC_C.Rust 9807_90deg.JPG")