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
sentence = ''


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


class App:
    global sentence

    def __init__(self, window, window_title, video_source=0):
        global textForm, text_file_num, sample_rate, test_df
        text_file_num = 1
        sample_rate = default_sample_rate
        window.resizable(False, False)
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        # open video source (by default this will try to open the computer webcam)
        self.vid = VideoFrame(self.video_source)
        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width=800, height=800)
        self.canvas.pack()
        # adding text promt and entry box
        self.txt_label = tkinter.Label(window, text="The translated text :")
        self.txt_label.place(x=50, y=490)
        self.txt_box = tkinter.Entry(
            window, justify=RIGHT, font=("Arial", 20, "bold"))
        self.txt_box.place(x=180, y=490, height=90, width=350)
        self.txt_box.configure(width=334)
        textForm = self.txt_box
        textForm.config(state=DISABLED)

        # adding buttons
        image = Image.open("icons/save.png")
        img = ImageTk.PhotoImage(image)
        self.save_but = tkinter.Button(window, text="save text", width=50, height=50, image=img,
                                       command=self.click_on_save)
        self.save_but.place(x=555, y=510)
        del_img = Image.open("icons/delete.png")
        del_img = del_img.resize((20, 20), Image.LANCZOS)
        img_del = ImageTk.PhotoImage(del_img)

        self.clear_but = tkinter.Button(
            window, image=img_del, command=self.clear_txt_box)
        self.clear_but.place(x=155, y=556)
        self.clean_label = tkinter.Label(window, text="Clear text")
        self.clean_label.place(x=145, y=580)

        self.save_label = tkinter.Label(window, text="Save Text")
        self.save_label.place(x=550, y=570)

        # Bind all keyboard pressed to keyPressed function.
        window.bind('<KeyPress>', self.keyPressed)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 1
        self.update()
        self.window.mainloop()

    def click_on_save(self):
        global textForm, text_file_num
        f = filedialog.asksaveasfile(mode='w', defaultextension=".txt")
        if f is None:  # asksaveasfile return `None` if dialog closed with "cancel".
            return
        data = textForm.get()
        data.encode(encoding="UTF-8", errors='strict')
        f.write(data)
        f.close()

    def clear_txt_box(self):
        global textForm
        textForm.config(state=NORMAL)
        textForm.delete(0, END)
        textForm.config(state=DISABLED)
        sentence = ''

    def exit_prog(self):
        self.window.destroy()

    #run the update method every frame
    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

        self.window.after(self.delay, self.update)
    
    def keyPressed(self, event):
        global test_df
        print(event.keycode)
        if event.keycode == 889192475:  # Escape
            self.window.destroy()
        elif event.keycode == 2063660802:  # Left
            self.vid.x0 = max((self.vid.x0 - 5, 0))
        elif event.keycode == 2113992448:  # Up
            self.viﬁﬁd.y0 = max((self.vid.y0 - 5, 0))
        elif event.keycode == 2080438019:  # Right
            self.vid.x0 = min(
                (self.vid.x0 + 5, self.vid.frame.shape[1] - self.vid.predWidth))
        elif event.keycode == 2097215233:  # Down
            self.vid.y0 = min(
                (self.vid.y0 + 5, self.vid.frame.shape[0] - self.vid.predWidth))
        elif event.keycode == 771752045:  # 'M' - Binary Mask
            self.vid.showMask = not self.vid.showMask
        elif event.keycode == 587202672:  # 'P' - Prediction
            self.vid.predict = not self.vid.predict

        elif event.keycode == 285212788:  # t - TestMode
            if self.vid.testMode:
                self.vid.testMode = False
                test_df.to_csv('test.csv', index=False)
                self.vid.predict = False
                print_csv_to_console()
            else:
                self.vid.testMode = True
                self.vid.predict = True
                test_df = pd.read_csv(test_csv_path)


