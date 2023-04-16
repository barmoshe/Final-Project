import cv2
import glob
import os
import random


def flipImages(rootDir='/Users/barmoshe/Final Project/TempImagesMerged', imgFormat=None):
    """
    Flip all images in given rootDirectory. (Make new copies).

    :param rootDir: the folder that contain sub-folders which with images. Ex: "Images/train".
    :param imgFormat: 'jpg' or 'png'. If none provided, both will be used.
    """
    if imgFormat is None:
        flipImages(rootDir, 'jpg')
        flipImages(rootDir, 'png')
    else:
        string = rootDir + "/*/*." + imgFormat
        filenames = glob.glob(string)
        if len(filenames) == 0:
            string = rootDir + "/*." + imgFormat
            filenames = glob.glob(string)
        for fileName in filenames:
            img = cv2.imread(fileName)
            if img is not None:
                flippedFilename = fileName.replace(
                    "." + imgFormat, "_flipped." + imgFormat)
                cv2.imwrite(filename=flippedFilename, img=cv2.flip(img, 1))
                print("Flipped " + fileName + "as " + flippedFilename)



def moveRandomFiles(from_dir, to_dir, percent):
    """
    Move random files between folders.

    :param from_dir: English path. Source directory.
    :param to_dir: English path. Destination directory.
    :param percent: Percent of files to move out of the source folder.
    """
    count = len(os.listdir(from_dir))
    numToMove = int(percent * count)
    try:
        os.makedirs(to_dir)
    except OSError as e:
        # print(e)
        pass

    for i in range(numToMove):  # [0,numToMove)
        fileName = random.choice(os.listdir(from_dir))
        os.rename(from_dir + "/" + fileName, to_dir + "/" + fileName)
        print("moved file " + fileName + " from " + from_dir + " to " + to_dir)

def moveProjectData(source="/Users/barmoshe/Final Project/TempImagesMerged", dest="/Users/barmoshe/Final Project/images/train", percent=1.0):
    """
    Move random data between test and train folders.  iterate subdirectories.

    :param source: English path. Source directory.
    :param dest: English path. Destination directory.
    :param percent: Percent of files to move out of the source folder.
    :raises: OSError if source folder does not exist.
    """
    for subdir in os.listdir(source):
        moveRandomFiles(from_dir=source + "/" + subdir,
                        to_dir=dest + "/" + subdir, percent=percent)


moveProjectData()
