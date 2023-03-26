
# W = Space
# X = Del
# Y = None
classes = 'A B C D E F G H I J K L M N O P Q R S T U V W X Y'.split()

# use this and not 'א'+index since includes finals.
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
    'ן': 'F',
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

test_csv_path = 'test.csv'
modelPath = 'Model/cnn12_model.h5'
modelWeights = 'Model/trainWeights.h5'