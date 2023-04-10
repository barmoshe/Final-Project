import platform as plat
import os
from PIL import ImageFont

# Get the platform (Windows, Linux, Darwin, etc.)
platform = plat.system()

# Define the classes (letters) used in the model
# 'W' represents a space, 'X' represents a deletion, and 'Y' represents nothing
classes = 'A B C D E F G H I J K L M N O P Q R S T U V W X Y'.split()

# Define the Hebrew alphabet (including spaces and final letters)
alphaBet = "א ב ג ד ה ו ז ח ט י כ ל מ נ ס ע פ צ ק ר ש ת".split()
alphaBet.append(' ')  # append space to be at the last index.
# finals: ך ם ן ף ץ

# Define a dictionary that maps Hebrew alphabet characters to their English counterparts
hebrew_to_english = {
    'א': 'A',
    'ב': 'B',
    'ג': 'C',
    'ד': 'D',
    'ה': 'E',
    'ו': 'F',
    'ז': 'G',
    'ח': 'H',
    'ט': 'I',
    'י': 'J',
    'כ': 'K',
    'ל': 'L',
    'מ': 'M',
    'נ': 'N',
    'ס': 'O',
    'ע': 'P',
    'פ': 'Q',
    'צ': 'R',
    'ק': 'S',
    'ר': 'T',
    'ש': 'U',
    'ת': 'V',
    'space': 'W',
    'del': 'X',
    'nothing': 'Y',
}

# Define the path to the test CSV file
test_csv_path = 'test.csv'

# Define the path to the model and its weights
modelPath = 'Model/cnn12_model.h5'
modelWeights = 'Model/trainWeights.h5'

# Define the image dimension (assumed to be square)
imgDim = 128

# Define key codes for some keyboard keys
key_codes = {}
key_codes[('t', 'Darwin')] = 285212788
key_codes[('t', 'Windows')] = 84
key_codes[('p', 'Darwin')] = 587202672
key_codes[('p', 'Windows')] = 80
key_codes[('m', 'Darwin')] = 771752045
key_codes[('m', 'Windows')] = 77
key_codes[('Esc', 'Darwin')] = 889192475
key_codes[('Esc', 'Windows')] = 27
key_codes[('Left', 'Darwin')] = 2063660802
key_codes[('Left', 'Windows')] = 37
key_codes[('Up', 'Darwin')] = 2113992448
key_codes[('Up', 'Windows')] = 38
key_codes[('Right', 'Darwin')] = 2080438019
key_codes[('Right', 'Windows')] = 39
key_codes[('Down', 'Darwin')] = 2097215233
key_codes[('Down', 'Windows')] = 40

# Define the font used for the text on the images
font = ImageFont.truetype(font='fonts/arial.ttf', size=32)
