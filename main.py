import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#Configuration file
config_file = 'Dataset/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
data_set = 'Dataset/frozen_inference_graph.pb'
# data_set1 ='resnet50_coco_best_v2.1.0.h5'

# Algorithm DNN using for train model 
model = cv2.dnn_DetectionModel(data_set, config_file)

#labels files 
classlabals = []
file_name = "Dataset/Labels.txt"
with open(file_name, 'rt') as fpt:
    classlabals = fpt.read().rstrip('\n').split('\n')

#model scale configuration
model.setInputSize(320, 320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)


#function 
def ImageDetect(input_image):

    font_scale = 3
    font = cv2.FONT_HERSHEY_PLAIN
    for ClassInd, cof, boxes in zip(classIndex.flatten(), confidece.flatten(), bbox):
        image_box = cv2.rectangle(input_img, boxes, (255, 0, 0), 2)
        image_put = cv2.putText(input_img, classlabals[ClassInd-1], (boxes[0]+10, boxes[1]+40),
                                font, fontScale=font_scale, color=(0, 0, 255), thickness=5)
        # final_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = classlabals[ClassInd - 1]
        st.write("Objects in image : " + label)

    return image_put



#title bar
st.title("Object Detection App")
st.write("Input Image")

# input image
file_img = st.sidebar.file_uploader(
    "Upload Your Photo", type=['jpg', 'jpeg', 'png'])


if file_img is None:
    st.write("Please any Image ")

else:
    st.image(file_img, width=500)
    input_img = Image.open(file_img)
    input_img = np.array(input_img)
    classIndex, confidece, bbox = model.detect((input_img), confThreshold=0.5)
    
    #function call 
    final_output = ImageDetect(input_img) 
    #output image
    st.write("Output image ")
    st.image(final_output, width=500)
    
    
    
#     for i in classIndex:
#         st.write()

    # img = cv2.imread(file_img)

    # gray = plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # ClassIndex, confidece, bbox = model.detect(file_img, confThreshold=0.5)

    # font_scale = 3
    # font = cv2.FONT_HERSHEY_PLAIN
    # for ClassInd, cof, boxes in zip(ClassIndex.flatten(), confidece.flatten(), bbox):
    #     cv2.rectangle(img, boxes, (255, 0, 0), 2)
    #     cv2.putText(img, classlabals[ClassInd-1], (boxes[0]+10, boxes[1]+40),
    #                 font, fontScale=font_scale, color=(0, 255, 0), thickness=3)
    #     output_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # st.image(output_img)
