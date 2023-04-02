import pandas as pd
import cv2
import glob
import os
import random
from globals import alphaBet


def print_csv_to_console():
    print_df = pd.read_csv('test.csv')
    print(print_df)


def finalizeHebrewString(hebrewString):
    """
    Convert hebrew string letters to finals if needed (After space).

    :param hebrewString: Hebrew sentence.
    :return: Valid hebrew sentence with final letters representation.
    """
    if type("hebrewString") is not str or len(hebrewString) == 0:
        return hebrewString
    hebrewString = hebrewString.replace('כ ', 'ך ')
    hebrewString = hebrewString.replace('מ ', 'ם ')
    hebrewString = hebrewString.replace('נ ', 'ן ')
    hebrewString = hebrewString.replace('פ ', 'ף ')
    hebrewString = hebrewString.replace('צ ', 'ץ ')
    hebrewString = hebrewString[:-1] + \
        convertHebrewLetterToFinal(hebrewString[-1])
    return hebrewString


def binaryMask(img):
    """
    Apply binary mask on raw rgb image.

    :param img: 3D np array.
    :return: processed image. (3D np array).
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (7, 7), 3)
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, new = cv2.threshold(
        img, 25, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return new


def convertEnglishToHebrewLetter(englishLetter):
    """
    Convert english letter to hebrew letter.

    :param englishLetter: English letter.
    :return: Hebrew letter.
    """
    if englishLetter == ' ' or englishLetter == 'w' or englishLetter == 'W':
        return ' '
    elif englishLetter == 'x' or englishLetter == 'X':
        return 'del'
    elif 'a' <= englishLetter <= 'v':
        return convertIndexToHebrewLetter(ord(englishLetter) - ord('a'))
    elif 'A' <= englishLetter <= 'V':
        return convertIndexToHebrewLetter(ord(englishLetter) - ord('A'))
    else:
        return ''


def convertIndexToHebrewLetter(index):
    """
    Convert index to hebrew letter.

    :param index: index in the range[0,23]. Out of range index will be converted to blank char.
    :return: Hebrew letter.
    """
    if index == 23:  # deletion
        return 'del'
    elif 0 <= index <= 22:  # 22 = space
        return alphaBet[index]
    else:
        return ''

def convertHebrewLetterToFinal(hebrewLetter):
    """
    Convert hebrew letter to final representation. Not will be changed if not convertable.

    :param hebrewLetter: Hebrew letter.
    :return: Final representation Hebrew letter.
    """
    if hebrewLetter == 'כ':
        return 'ך'
    elif hebrewLetter == 'מ':
        return 'ם'
    elif hebrewLetter == 'נ':
        return 'ן'
    elif hebrewLetter == 'פ':
        return 'ף'
    elif hebrewLetter == 'צ':
        return 'ץ'
    else:
        return hebrewLetter