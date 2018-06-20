import csv
import os
import sys
from sendemail import sendemail
_DATA_DIR = "segments_train/"
_OUT_DIR = "big_split_segments_train/"
snippets_video = []
snippets_time = []


with open('AVA2.1/ava_mini_split_train_big.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        snippets_video.append(row[0])
        snippets_time.append(row[1])

#Find unique segments
        
currentVideo = '';
currentTime = '';
for i in range(len(snippets_video)):
    if i%100 == 0:
        print("Video #" + str(i) + " of " +str(len(snippets_video)) + ".")
    if currentVideo == snippets_video[i] and currentTime == snippets_time[i]:
        pass
    else:
        currentVideo = snippets_video[i]
        currentTime = snippets_time[i]
        #print(currentTime)
        
        if currentTime[0] == "0":
            currentTimeStr = currentTime[1:]
        else:
            currentTimeStr = currentTime # Fixed this :D
        output_folder = currentVideo + "_" + currentTimeStr
        videoPath = _DATA_DIR + currentVideo + "_" + currentTimeStr + ".avi"
        if not os.path.exists(videoPath):
            print(videoPath + " not found")
            sys.exit(0)
        else:
            if not os.path.exists(_OUT_DIR + output_folder):
                os.makedirs(_OUT_DIR + output_folder)
                cmd = "ffmpeg -i " + videoPath + " -vf select='eq(n\,25)+eq(n\,35)+eq(n\,45)+eq(n\,55)+eq(n\,65)' -vsync 0 -s 224x224 " + _OUT_DIR + output_folder +  "/frames%d.jpg"
                os.system(cmd)
            
#sendemail(from_addr    = 'pythonscriptsisr@gmail.com', 
#          to_addr_list = ['pedro_abreu95@hotmail.com','joaogamartins@gmail.com'],
#          cc_addr_list = [], 
#          subject      = 'Extract and Resize', 
#          message      = 'The function is FCKING DONE!', 
#          login        = 'pythonscriptsisr@gmail.com', 
#          password     = '1!qwerty')