import numpy as np
import pandas as pd
import cv2
import glob
import os
import random
from globals import alphaBet


def print_csv_to_console():
    """
    Read and print the contents of a CSV file to the console.
    """
    # Load the CSV file into a pandas dataframe
    csv_df = pd.read_csv('test.csv')

    # Print the contents of the dataframe to the console
    print(csv_df)


def finalizeHebrewString(hebrewString):
    """
    Convert Hebrew string letters to finals if needed (After space).

    :param hebrewString: Hebrew sentence.
    :return: Valid Hebrew sentence with final letters representation.
    """
    # Check if the input is a non-empty string
    if not isinstance(hebrewString, str) or len(hebrewString) == 0:
        return hebrewString

    # Replace letters that have final forms with their final form
    hebrewString = hebrewString.replace('כ ', 'ך ')
    hebrewString = hebrewString.replace('מ ', 'ם ')
    hebrewString = hebrewString.replace('נ ', 'ן ')
    hebrewString = hebrewString.replace('פ ', 'ף ')
    hebrewString = hebrewString.replace('צ ', 'ץ ')

    # Convert the last letter of the string to its final form
    hebrewString = hebrewString[:-1] + \
        convertHebrewLetterToFinal(hebrewString[-1])

    # Return the modified string
    return hebrewString


def binaryMask(img):
    """
    Apply binary mask on raw RGB image.

    :param img: 3D numpy array representing an RGB image.
    :return: Processed image as a 3D numpy array.
    """
    # Convert the RGB image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to smooth the image
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)

    # Apply adaptive thresholding to create a binary image
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Apply Otsu's thresholding to further refine the binary image
    ret, final = cv2.threshold(
        binary, 25, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Return the final binary image
    return final


def convertEnglishToHebrewLetter(englishLetter):
    """
    Convert an English letter to its Hebrew equivalent.

    :param englishLetter: A single English letter.
    :return: The corresponding Hebrew letter, or an empty string if there is no mapping for the input letter.
    """
    if englishLetter == ' ' or englishLetter == 'w' or englishLetter == 'W':
        # If the input letter is a space or 'w'/'W', return a space character.
        return ' '
    elif englishLetter == 'x' or englishLetter == 'X':
        # If the input letter is 'x'/'X', return the Hebrew equivalent of 'delete'.
        return 'del'
    elif 'a' <= englishLetter <= 'v':
        # If the input letter is between 'a' and 'v', convert its index to a Hebrew letter using the `convertIndexToHebrewLetter` function.
        return convertIndexToHebrewLetter(ord(englishLetter) - ord('a'))
    elif 'A' <= englishLetter <= 'V':
        # If the input letter is between 'A' and 'V', convert its index to a Hebrew letter using the `convertIndexToHebrewLetter` function, but in uppercase.
        return convertIndexToHebrewLetter(ord(englishLetter) - ord('A')).upper()
    else:
        # If the input letter is not one of the above cases, return an empty string.
        return ''


def convertIndexToHebrewLetter(index):
    """
    Convert an index to a corresponding Hebrew letter.

    :param index: An integer index in the range [0, 23] representing a Hebrew letter.
                  Indices outside this range are converted to a blank string.
    :return: A string containing the corresponding Hebrew letter or blank string.
    """
    if index == 23:  # the index represents a deletion
        return 'del'
    elif 0 <= index <= 22:  # the index represents a valid Hebrew letter
        return alphaBet[index]
    else:  # the index is out of range
        return ''


def convertHebrewLetterToFinal(hebrewLetter):
    """
    Convert a Hebrew letter to its final representation, if applicable.

    :param hebrewLetter: A single character string representing a Hebrew letter.
    :return: A single character string representing the final representation of the input Hebrew letter.
             If the input is not convertible, the function returns the input letter unchanged.
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
    else:  # the input is not convertible
        return hebrewLetter


def print_csv_to_console():
    # Read the contents of 'test.csv' into a DataFrame
    print_df = pd.read_csv('test.csv')

    # Print the DataFrame to the console
    print(print_df)


def write_to_csv(prediction, rightprediction):
    """
    Write a new row to a global pandas DataFrame and return it.

    Parameters:
    prediction (str): the predicted value.
    rightprediction (str): the actual value.

    Returns:
    test_df (pandas.DataFrame): the updated DataFrame.
    """
    global test_df

    # Get the current time
    current_time = datetime.datetime.now()

    # Create a new dictionary with the row data
    new_row = {
        'prediction': prediction,
        'right_prediction': rightprediction,
        'time': current_time
    }

    # Add the new row to the DataFrame and reset the index
    test_df = pd.concat([test_df, pd.DataFrame(
        new_row, index=[0])], ignore_index=True)

    # Return the updated DataFrame
    return test_df
