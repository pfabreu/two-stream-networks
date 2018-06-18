import glob
import os
from sendemail import sendemail
_DATA_DIR = "/media/pedro/actv4/AVA/segments_test/"
_OUT_DIR = "/media/pedro/actv4/AVA/rgb_test/"


for v in glob.glob(_DATA_DIR + "*"):

    vname = v.split("/")[-1]
    output_name = vname[:-4]
    videoPath = _DATA_DIR + vname
    if not os.path.exists(videoPath):
        print(videoPath + " not found")
    else:
        if not os.path.exists(_OUT_DIR + output_name):
            os.makedirs(_OUT_DIR + output_name)
        print _OUT_DIR + output_name
        cmd = "ffmpeg -loglevel panic -i " + videoPath + " -vf select='eq(n\,25)+eq(n\,35)+eq(n\,45)+eq(n\,55)+eq(n\,65)' -vsync 0 -s 224x224 " + _OUT_DIR + output_name + "/frames%d.jpg"
        os.system(cmd)

sendemail(from_addr='pythonscriptsisr@gmail.com',
          to_addr_list=['pedro_abreu95@hotmail.com', 'joaogamartins@gmail.com'],
          cc_addr_list=[],
          subject='Extract rgb and resize (for the test)',
          message='The function is FCKING DONE!',
          login='pythonscriptsisr@gmail.com',
          password='1!qwerty')
