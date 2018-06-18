import csv
import os

set_type = "validation"

_DATA_DIR = "/media/pedro/actv4/AVA-split/" + set_type + "/rgb/"

i = 0

print "Checking if all folders exist"
with open('ava_mini_split_' + set_type + '.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        vidname = row[0] + "_" + row[1].lstrip("0")
        if not os.path.exists(_DATA_DIR + vidname):
            # Save them to a file
            print _DATA_DIR + vidname
        i += 1


i = 0
print "Checking if all folders have correct number of files"
for dirs in os.listdir(_DATA_DIR):
    num_files = len(os.listdir(_DATA_DIR + dirs))
    if num_files < 5:
        print dirs + " has " + str(num_files)
        i += 1
print i