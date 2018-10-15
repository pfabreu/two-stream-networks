import cv2
import np_opencv_module as npcv
import numpy as np
from yolt_python import LaplacianBlending as fv
from matplotlib import pyplot as plt
from random import randint
import csv
import os
from PIL import Image
import gc

# Load ava CSV
snippets_video = []
snippets_time = []
snippets_bb = []
set_type = "train"
datadir = '/media/pedro/actv-ssd/segments_' + set_type + '/'
outdir = '/media/pedro/actv-ssd/foveanew_' + set_type + '/'
imres = np.array([224, 224])
with open('AVA_' + set_type.title() + '_Custom_Corrected.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        snippets_video.append(row[0])
        snippets_time.append(row[1])
        snippets_bb.append(np.array([float(row[2]), float(row[3]), float(row[4]), float(row[5])]))  # xmin,ymin,xmax,ymax

current_video = ''
current_time = ''
current_bb = np.array([-1, -1, -1, -1])

for frame in range(len(snippets_video)):
    if (frame % 100 == 0):
        print "Now on frame #" + str(frame) + " of " + str(len(snippets_video)) + ".\n"
    # Check if current frame is a new one:
    if (snippets_video[frame] == current_video) and \
        (snippets_time[frame] == current_time) and \
            (sum(current_bb == snippets_bb[frame]) == len(current_bb)):
        # This is the same bb with a diff action. skip.
        pass
    else:
        current_video = snippets_video[frame]
        current_time = snippets_time[frame]
        if current_time[0] == '0':
            current_timestr = current_time[1:]
        else:
            current_timestr = current_time
        current_bb = snippets_bb[frame]  # xmin,ymin,xmax,ymax
        # calculate the center of the bb
        xcenter = imres[0] * (current_bb[0] + (current_bb[2] - current_bb[0]) / 2.0)
        ycenter = imres[1] * (current_bb[1] + (current_bb[3] - current_bb[1]) / 2.0)
        center = [int(xcenter), int(ycenter)]
        # sigma will be half of the size of the largest dimension of the bounding box.
        sigma_x = int(0.5 * imres[0] * (current_bb[2] - current_bb[0]))
        sigma_y = int(0.5 * imres[1] * (current_bb[3] - current_bb[1]))
#        if bbhsize > bbvsize:
#            sigma = bbhsize
#        else:
#            sigma = bbvsize
#        sigma = int(sigma)
#        #load the 5 images and foveate them.
        viddir = datadir + current_video + '_' + current_timestr + '.avi'
        outdirvidkey = outdir + current_video + '_' + current_timestr + '_' + str(current_bb[0]) + '_' + str(current_bb[1]) + '_' + str(current_bb[2]) + '_' + str(current_bb[3])
        if not os.path.isfile(viddir):
            print "cant find " + viddir
        else:
            if not os.path.exists(outdirvidkey):
                # Make output directory
                # print outdirvidkey
                os.makedirs(outdirvidkey)
            missing = False
            for fc in range(1, 6):
                if not os.path.exists(outdirvidkey + "/frames" + str(fc) + ".jpg"):
                    missing = True
                    break
            if missing is True:
                # Load original_video and get the BB
                vidcap = cv2.VideoCapture(viddir)
                success, img = vidcap.read()
                height, width, layers = img.shape
                # Loop through video, for the desired 5 frames compute fovea and save images
                f = 1
                fcount = 1
                while success:
                    success, img = vidcap.read()

                    if success is False:
                        vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        break
                    if f in [25, 35, 45, 55, 65]:
                        # print img.type
                        #my_mat_img = npcv.test_np_mat(img)
                        #m2 = cv2.CreateMat(height, width, cv2.CV_32FC3)
                        #m0 = cv2.fromarray(img)
                        #cv2.CvtColor(m0,m2, cv2.CV_GRAY2BGR)
                        my_lap_obj = fv(width, height, 4, sigma_x, sigma_y)
                        ctr = npcv.test_np_mat(np.array(center))
                        foveated_img = my_lap_obj.foveate(img, ctr)

                        foveated_img = cv2.resize(foveated_img, (224, 224), interpolation=cv2.INTER_NEAREST)
                        # BGR to RGB
                        rgb_img = foveated_img[:, :, ::-1].copy()
                        if not os.path.exists(outdirvidkey + "/frames" + str(fcount) + ".jpg"):
                            Image.fromarray(rgb_img).save(outdirvidkey + "/frames" + str(fcount) + ".jpg")
                        ctr = None
                        # Save image
                        fcount += 1
                    f += 1

                vidcap.release()
                cv2.destroyAllWindows()
