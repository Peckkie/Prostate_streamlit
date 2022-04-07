# import for streamlit
import streamlit as st
import tensorflow as tf
import streamlit as st
# import for model
import json 
import pandas as pd
import numpy as np
from pathlib import Path
import PIL
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
import cv2 
import torch
import tqdm

# Load model weight
@st.cache(allow_output_mutation=True)
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='model/ModelR1_Prostate5m_last.pt', force_reload=True) # or yolov5m, yolov5l, yolov5x, custom
    return model
with st.spinner('Model is being loaded..'):
    model=load_model()

st.write("""
         # 🔗 Prostate Detection
         """
         )


def predict_box(path_img, thres): ## **-- path/to/image/.jpg 
    ##เตรียม Data เข้าสู้ Network
    image_cv = cv2.imread(path_img)[..., ::-1] 
    model.conf = thres # NMS confidence threshold
    model.iou = 0.50 # NMS IoU threshold
    model.classes = None   # (optional list) filter by class, i.e. = [0, 15, 16] for persons, cats and dogs
    model.multi_label = False  # NMS multiple labels per box
    model.max_det = 10
    ### Result ---*** 
    results = model(image_cv, size=640)  
    all_table = [] 
    for i in range(len(results.xyxy)):
        if len(results.xyxy[i]) == 0 :
            #print(i)
            results.xyxy[i] = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float)
        table = results.pandas().xyxy[i]
        table['Path'] = path_img  #path/to/image
        table['image_id'] = path_img.split('/')[-1]
        all_table.append(table)

    con_table = pd.concat(all_table,axis=0).reset_index(drop=True)
    return con_table

def plot_img(df):
    img_path = list(set(df['Path']))
    img_c = cv2.imread(img_path[0])
    #label = str(df['confidence'][0])
    if df['confidence'][0] == 0:
        image_pre_ = img_c
        #string = "[Not Detected]"
        #st.success(string)
    else:
        for j in range(len(df)):
            xmin_pre = int(df['xmin'][j])
            ymin_pre = int(df['ymin'][j])
            xmax_pre = int(df['xmax'][j])
            ymax_pre = int(df['ymax'][j])
            #label = df['name'][j]+':conf '+str((df['confidence'][j]* 1e4).astype(int) / 1e4)
            #label = str((df['confidence'][j]* 1e3).astype(int) / 1e3)
            label = 'B'+str(j+1)+' : '+str((df['confidence'][j]* 1e4).astype(int) / 1e4)
            #label_ = 'B'+str(j+1)
            #conf_box = (df['confidence'][j]* 1e4).astype(int) / 1e4
            #st.write(f'{label}: Postate Cancer, Confident {conf_box}')
            #string = f'{label_}: Cervical fracture, Confident {conf_box}'
            #st.success(string)
            image_pre = cv2.rectangle(img_c, (xmin_pre ,ymin_pre), (xmax_pre, ymax_pre), (32, 32, 216), 3) #32, 32, 216 | 57, 0, 199
            image_pre_ = cv2.putText(image_pre, label, (xmin_pre, ymin_pre-10), 3, 0.8, [32, 32, 216], thickness=2, lineType=1)
        path_img = "image/"+file.name
        cv2.imwrite(path_img, image_pre_)
    
def plot_img_str(df):
    img_path = list(set(df['Path']))
    img_c = cv2.imread(img_path[0])
    if df['confidence'][0] == 0:
        image_pre_ = img_c
        string = "[Non't Detected]"
        st.success(string)
    else:
        for j in range(len(df)):
            label_ = 'B'+str(j+1)
            conf_box = (df['confidence'][j]* 1e4).astype(int) / 1e4
            string = f'{label_}: Postate Cancer, Confident {conf_box}'
            st.success(string)

file = st.file_uploader("Please upload an image file", type=["jpg", "png", ".jpeg"])
import cv2
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)

thres = st.slider("Confidence threshold", 0.00, 1.00, 0.70)

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    #st.image(image, use_column_width=True)
    image.save("image/"+file.name)
    
    path_img = 'image/'+file.name
    #st.write(file.type +'/'+file.name) #file.name
    df = predict_box(path_img, thres)
    plot_img(df)
    st.image(path_img, use_column_width=True)
    plot_img_str(df)
