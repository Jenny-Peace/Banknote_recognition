import streamlit as st
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
#from tensorflow.keras.preprocessing.image import ImageDataGenerator

menu = ['Home','Upload Your Photo','Capture From Webcam','Ask Me ~~~']

choice = st.sidebar.selectbox('Check your money with options below:', menu)


#Load your model and check create the class_names list
Model_Path = 'my_model_checkpoint_DenseNet_Nov23.h5'

class_names = ['1,000', '10,000', '100,000',  '2,000',  '20,000',  '200,000',  '5,000', '50,000',  '500,000']
model = tf.keras.models.load_model(Model_Path)
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

st.header("Let My First Web App Help You!")
if choice=='Home':
    st.title("Wana scan your money??? ")

    st.write("Enjoy some music first")
    st.write("")
    st.video('https://www.youtube.com/watch?v=ETxmCCsMoD0')

    st.write("Money, money, money")
    st.write("Must be funny!!!")
    st.balloons()
    

if choice == 'Upload Your Photo':
    st.title('Upload Your Photo')
    photo_uploaded = st.file_uploader('Please take a look at types of the file', ['png', 'jpeg', 'jpg'])
    if photo_uploaded!=None:
        image_np = np.asarray(bytearray(photo_uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(image_np, 1)
        st.image(img, channels='BGR')

        #st.write(photo_uploaded.size)
        #st.write(photo_uploaded.type)

        #Resize the Image according with your model
        img = cv2.resize(img,(224,224),interpolation = cv2.INTER_AREA)
        #Expand dim to make sure your img_array is (1, Height, Width , Channel ) before plugging into the model
        img_array  = np.expand_dims(img, axis=0)
        
        #JENNY code
        prediction = model.predict(img_array)
        a = np.argmax(prediction,axis=1)
        #st.write(a[0])
        result = class_names[int(a)]
        st.write('The denomination is:',result)
        

elif choice == 'Capture From Webcam':
    st.title('Capture From Webcam')
    cap = cv2.VideoCapture(0)  # device 0
    run = st.checkbox('Show Your Webcam')
    capture_button = st.checkbox('Capture now!')

    captured_image = np.array(None)


    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    FRAME_WINDOW = st.image([])
    while run:
        ret, frame = cap.read()        
        # Display Webcam
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB ) #Convert color
        FRAME_WINDOW.image(frame)

        if capture_button:      
            captured_image = frame
            break

    cap.release()

    if  captured_image.all() != None:
        st.write('Image is captured:')
        st.image(captured_image)

        #Resize the Image according with your model
        captured_image = cv2.resize(captured_image,(224,224),interpolation = cv2.INTER_AREA)
        #Expand dim to make sure your img_array is (1, Height, Width , Channel ) before plugging into the model
        img_array  = np.expand_dims(captured_image, axis=0)
        
        #JENNY code
        prediction = model.predict(img_array)
        a = np.argmax(prediction,axis=1)
        #st.write(a[0])
        result = class_names[int(a)]
        st.write('The denomination is:',result)
        

elif choice=='Ask Me ~~~':
    st.title('Ask Me Now!')
    st.success('Contact me at Discord Jenny Peace#2703')
    
    st.image('https://c.tenor.com/LwS8qVjM3rQAAAAC/dino-wonder.gif',
    caption="I am a wonder girl who wonder about EVERYTHING",
            use_column_width='auto')
    
    st.balloons()

