import pandas as pd


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
