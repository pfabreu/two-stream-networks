import csv
import cv2
import os
import numpy as np
from PIL import Image, ImageFilter

set_type = "train"
_DATA_DIR = "/media/pedro/actv-ssd/segments_" + set_type + "/"
_OUT_DIR = "/media/pedro/actv-ssd/gaussnew_" + set_type + "/"

snippets_video = []
snippets_time = []
snippets_bbs = []


def array2PIL(arr, size):
    mode = 'RGBA'
    arr = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2])
    if len(arr[0]) == 3:
        arr = numpy.c_[arr, 255 * numpy.ones((len(arr), 1), numpy.uint8)]
    return Image.frombuffer(mode, size, arr.tostring(), 'raw', mode, 0, 1)

# TODO Remember to change the file name here too
with open('src/python_bindings/AVA_' + set_type.title() + '_Custom_Corrected.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        snippets_video.append(row[0])
        snippets_time.append(row[1])
        snippets_bbs.append(row[2] + "_" + row[3] + "_" + row[4] + "_" + row[5])

currentVideo = ''
currentTime = ''
currentbbs = ''
sigma = 10
for i in range(len(snippets_video)):
    if i % 100 == 0:
        print("Video #" + str(i) + " of " + str(len(snippets_video)) + ".")
    if currentVideo == snippets_video[i] and currentTime == snippets_time[i] and currentbbs == snippets_bbs[i]:
        pass
    else:
        currentVideo = snippets_video[i]
        currentTime = snippets_time[i]
        currentbbs = snippets_bbs[i]
        if currentTime[0] == "0":
            currentTimeStr = currentTime[1:]
        else:
            currentTimeStr = currentTime
        output_folder = _OUT_DIR + currentVideo + "_" + currentTimeStr + "_" + currentbbs
        videoPath = _DATA_DIR + currentVideo + "_" + currentTimeStr + ".avi"
        if not os.path.exists(videoPath):
            print(videoPath + " not found")
        else:
            if not os.path.exists(output_folder):
                # Make output directory
                os.makedirs(output_folder)
                # Load original_video and get the BB
                vidcap = cv2.VideoCapture(videoPath)
                b = currentbbs.split('_')
                success, img = vidcap.read()
                height, width, layers = img.shape
                bx = (int(float(b[0]) * width), int(float(b[1]) * height),
                      int(float(b[2]) * width), int(float(b[3]) * height))
                # Loop through video, for the desired 5 frames compute gaussian and save images
                f = 1
                fcount = 1
                while success:
                    success, img = vidcap.read()

                    if success is False:
                        vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        break
                    if f in [25, 35, 45, 55, 65]:
                        # print img.shape
                        imga = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(imga)
                        box = img.crop(bx)
                        # Blur the whole image
                        img = img.filter(ImageFilter.GaussianBlur(sigma))
                        img.paste(box, bx)  # Paste them back
                        img = img.resize((224, 224), Image.NEAREST)
                        img.save(output_folder + "/frames" + str(fcount) + ".jpg")
                        # Save image
                        fcount += 1
                    f += 1

                vidcap.release()
                cv2.destroyAllWindows()
