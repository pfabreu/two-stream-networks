import csv
import os
import sys
import glob
from utils import sendemail
_DATA_DIR = "segments_test/"
_OUT_DIR = "test/rgb/"


#Find unique segments
counter = 1;
for v in glob.glob(_DATA_DIR+"*"):
	print(counter)         
	counter += 1;
	if not os.path.exists(v):
		print(v + " not found")
	else:
		output_folder = v.rsplit("/",1)[1];
		output_folder = output_folder[:-4]
		#print(output_folder)
		if not os.path.exists(_OUT_DIR + output_folder):
			os.makedirs(_OUT_DIR + output_folder)
			cmd = "ffmpeg -loglevel -8 -i " + v + " -vf select='eq(n\,25)+eq(n\,35)+eq(n\,45)+eq(n\,55)+eq(n\,65)' -vsync 0 -s 224x224 " + _OUT_DIR + output_folder +  "/frames%d.jpg"
			os.system(cmd)
			
sendemail(from_addr    = 'pythonscriptsisr@gmail.com', 
		  to_addr_list = ['pedro_abreu95@hotmail.com','joaogamartins@gmail.com'],
		  cc_addr_list = [], 
		  subject      = 'Extract and Resize', 
		  message      = 'The function is FCKING DONE!', 
		  login        = 'pythonscriptsisr@gmail.com', 
		  password     = '1!qwerty')