import csv
import cv2
import os
from PIL import Image, ImageFilter

set_type = "test"
_DATA_DIR = "/media/pedro/actv-ssd/segments_" + set_type + "/"
_OUT_DIR = "/media/pedro/actv-ssd/rgbnew_" + set_type + "/"

snippets_video = []
snippets_time = []
snippets_bbs = []

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
                # Loop through video, for the desired 5 frames compute gaussian and save images
                f = 1
                fcount = 1
                while success:
                    success, img = vidcap.read()

                    if success is False:
                        vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        break
                    if f in [25, 35, 45, 55, 65]:
                        imga = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(imga)
                        img = img.resize((224, 224), Image.NEAREST)
                        img.save(output_folder + "/frames" + str(fcount) + ".jpg")
                        # Save image
                        fcount += 1
                    f += 1

                vidcap.release()
                cv2.destroyAllWindows()
