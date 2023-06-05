import asyncio
import copy
import datetime
import tkinter
import numpy as np
import pandas as pd
import cv2
from tkinter import *
from tkinter import Tk, filedialog
from tkinter.ttk import Label
from PIL import Image, ImageTk, ImageDraw
from keras.models import load_model
from globals import *
from utilities import *
from tkinter import ttk

# Globals
global text_file_num, e1, sample_rate, glob_root
model = load_model(modelPath)
model.load_weights(modelWeights)
dataColor = (0, 255, 0)
pred = ''
prevPred = ''
sentence = ""
default_sample_rate = 6
count = default_sample_rate
threshold = 0.8
isOn = False
current_selection = None
count1 = 0
test_df = pd.read_csv(test_csv_path)
sentence = ''


def write_to_csv(prediction, rightprediction):
    global test_df

    current_time = datetime.datetime.now()

    new_row = {
        'prediction': prediction,
        'right_prediction': rightprediction,
        'time': current_time
    }

    test_df = pd.concat([test_df, pd.DataFrame(
        new_row, index=[0])], ignore_index=True)

    return test_df


async def predictImg(roi, test_mode=False):
    global count, sentence
    global pred, prevPred, textForm

    img = cv2.resize(roi, (imgDim, imgDim))
    img = np.float32(img) / 255.
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    vec = model.predict(img)
    pred = convertEnglishToHebrewLetter(classes[np.argmax(vec[0])])
    maxVal = np.amax(vec)

    if maxVal < threshold or pred == '':
        pred = ''
        count = sample_rate
    elif pred != prevPred:
        prevPred = pred
        count = sample_rate
    else:
        count = count - 1
        if count == 0:
            count = sample_rate
            if test_mode:
                cv2.imwrite("lastRoi.jpg", cv2.cvtColor(
                    roi, cv2.COLOR_RGB2BGR))
                if isOn == False:
                    show_test_popup("lastRoi.jpg")
            else:
                if pred == 'del':
                    sentence = sentence[:-1]
                else:
                    sentence = sentence + pred
                if pred == ' ':
                    pred = 'space'
                textForm.config(state=NORMAL)
                textForm.delete(0, END)
                textForm.insert(0, (finalizeHebrewString(sentence)))
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
        self.vid = VideoFrame(self.video_source)
        self.canvas = tkinter.Canvas(window, width=800, height=800)
        self.canvas.pack()
        self.txt_label = tkinter.Label(window, text="The translated text :")
        self.txt_label.place(x=50, y=490)
        self.txt_box = tkinter.Entry(
            window, justify=RIGHT, font=("Arial", 20, "bold"))
        self.txt_box.place(x=180, y=490, height=90, width=350)
        self.start_button = tkinter.Button(
            window, text="Start", font=("Arial", 20, "bold"), command=self.start_recognition)
        self.start_button.place(x=570, y=300, height=80, width=170)
        self.stop_button = tkinter.Button(
            window, text="Stop", font=("Arial", 20, "bold"), command=self.stop_recognition)
        self.stop_button.place(x=570, y=400, height=80, width=170)
        self.load_button = tkinter.Button(
            window, text="Load Video", font=("Arial", 20, "bold"), command=self.load_video)
        self.load_button.place(x=570, y=500, height=80, width=170)
        self.save_button = tkinter.Button(
            window, text="Save", font=("Arial", 20, "bold"), command=self.save_text)
        self.save_button.place(x=570, y=600, height=80, width=170)
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def start_recognition(self):
        global isOn, pred, prevPred, count, sentence
        isOn = True
        pred = ''
        prevPred = ''
        count = default_sample_rate
        sentence = ''
        self.txt_box.delete(0, END)
        self.txt_box.config(state=NORMAL)
        self.txt_box.insert(0, '')
        self.txt_box.config(state=DISABLED)
        self.vid.start_recognition()

    def stop_recognition(self):
        global isOn
        isOn = False
        self.vid.stop_recognition()

    def load_video(self):
        file = filedialog.askopenfilename(
            initialdir='/', title='Select Video File', filetypes=(("Video Files", "*.mp4"), ("All Files", "*.*")))
        if file:
            self.vid.load_video(file)

    def save_text(self):
        global text_file_num, test_df
        filename = f"test_output_{text_file_num}.csv"
        test_df.to_csv(filename, index=False)
        text_file_num += 1
        print(f"Test output saved to {filename}")

    def on_closing(self):
        self.vid.release()
        self.window.destroy()


class VideoFrame:
    def __init__(self, video_source=0):
        self.video_source = video_source
        self.vid = cv2.VideoCapture(self.video_source)
        self.vid.set(3, 800)
        self.vid.set(4, 800)
        self.width = self.vid.get(3)
        self.height = self.vid.get(4)
        self.window = None
        self.canvas = None
        self.photo = None
        self.recognizer = sr.Recognizer()
        self.stream = None

    def start_recognition(self):
        self.stream = Thread(target=self.recognize_speech)
        self.stream.start()

    def stop_recognition(self):
        self.stream.join()

    def load_video(self, video_file):
        self.vid = cv2.VideoCapture(video_file)
        self.width = self.vid.get(3)
        self.height = self.vid.get(4)

    def recognize_speech(self):
        global isOn, pred, prevPred, count, sentence
        while isOn:
            ret, frame = self.vid.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if count == sample_rate:
                pred = self.predict(gray)
                if pred != prevPred:
                    sentence += pred + ' '
                    prevPred = pred
                count = 0

            count += 1

            cv2.putText(frame, pred, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            self.photo = PIL.ImageTk.PhotoImage(
                image=PIL.Image.fromarray(frame))
            if self.canvas is None:
                self.canvas = tkinter.Canvas(
                    app.window, width=self.width, height=self.height)
                self.canvas.place(x=0, y=0)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

    def predict(self, frame):
        # Implement your prediction logic here
        # This is just a placeholder
        return "Predicted Text"


# Create an instance of the App class
window = tkinter.Tk()
app = App(window, "Speech Recognition App")
window.mainloop()
