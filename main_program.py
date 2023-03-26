"""
Main application. The GUI of the software.

"""
"""
Main application. The GUI of the software.

"""

# Importing necessary libraries and modules


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Globals
import asyncio  # For handling asynchronous programming
import copy  # For making deep copies of objects
import time  # For working with time-related functions
import tkinter  # For creating graphical user interfaces (GUIs)
from tkinter import *  # Importing all functions from tkinter
from tkinter import Tk, filedialog  # Specific tkinter modules
from tkinter.ttk import Label  # More tkinter modules for GUIs
import numpy as np  # For numerical computing
from PIL import Image, ImageTk  # For working with images
from PIL import ImageDraw  # For drawing on images
from keras.models import load_model  # For loading machine learning models
from globals import *  # Custom module for project parameters
from utilities import *  # Custom module for utility functions
from tkinter import ttk  # More tkinter modules for GUIs
import datetime  # For working with dates and times
import pandas as pd  # For working with data in a table format
global text_file_num, e1, sample_rate, glob_root

model = load_model(modelPath)
model.load_weights(modelWeights)
dataColor = (0, 255, 0)
pred = ''
prevPred = ''
sentence = ""
default_sample_rate = 9
count = default_sample_rate
threshold = 0.8  # the threshold for the prediction  Between 0 and 1
isOn = False
current_selection = None
count1 = 0
test_df = pd.read_csv(test_csv_path)


async def predictImg(roi, test_mode=False):
    """
    Asynchronously prediction.

    :param roi: preprocessed image.
    """
    global count, sentence
    global pred, prevPred, textForm

    # Resizing the input image to a fixed size
    img = cv2.resize(roi, (imgDim, imgDim))

    # Converting the image to a floating-point representation and scaling it to values between 0 and 1
    img = np.float32(img) / 255.

    # Adding a new dimension to the image to represent the channel dimension
    img = np.expand_dims(img, axis=-1)

    # Adding a new dimension to the image to represent the batch size dimension
    img = np.expand_dims(img, axis=0)

    # The 'predict' function of the model takes in the preprocessed image as input and returns a prediction vector.
    # This vector contains the predicted probability of the image belonging to each class in the classification task.
    vec = model.predict(img)

    # get the hebrew letter of the prediction from the dictionary
    pred = convertEnglishToHebrewLetter(classes[np.argmax(vec[0])])

    # We use the 'amax' function from numpy to get the maximum value in the 'vec' array.
    # This is useful for determining which class the model predicts the input image belongs to.
    maxVal = np.amax(vec)

    # If the maximum value is greater than the threshold, we consider the prediction to be valid.
    if maxVal < threshold or pred == '':
        pred = ''  # reset the prediction
        count = sample_rate  # reset the counter
    elif pred != prevPred:
        prevPred = pred  # update the previous prediction
        count = sample_rate  # reset the counter
    else:  # maxVal >= Threshold && pred == prevPred
        count = count - 1
        if count == 0:
            count = sample_rate  # reset the counter
            if test_mode:
                cv2.imwrite("lastRoi.jpg", cv2.cvtColor(
                    roi, cv2.COLOR_RGB2BGR))
                if isOn == False:
                    show_popup("lastRoi.jpg")
            else:
                if pred == 'del':
                    sentence = sentence[:-1]
                else:
                    sentence = sentence + pred
                if pred == ' ':
                    pred = 'space'
                # We set the 'state' attribute of the 'textForm' object to 'NORMAL'.
                # This allows us to modify the contents of the text box.
                textForm.config(state=NORMAL)

                # We delete the current contents of the text box.
                textForm.delete(0, END)

                # We insert the 'sentence' variable into the text box.
                # We use the 'finalizeHebrewString' function to ensure that the string is displayed correctly in Hebrew.
                textForm.insert(0, (finalizeHebrewString(sentence)))

                # We set the 'state' attribute of the 'textForm' object to 'DISABLED'.
                # This prevents the user from modifying the contents of the text box.
                textForm.config(state=DISABLED)
