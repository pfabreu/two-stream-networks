import cv2
import numpy as np
import np_opencv_module as npcv
from yolt_python import LaplacianBlending as fv
from matplotlib import pyplot as plt
from random import randint
import time

center=[230, 150];
sigma_x_max=200
sigma_y_max=100
levels=10
img = cv2.imread('image.jpg')
img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
height, width, channels = img.shape

sigma_x=sigma_x_max
sigma_y=sigma_y_max

# Create the Laplacian blending object
my_lap_obj=fv(width,height,levels,sigma_x,sigma_y)
try:
    while True:
        start = time.time()

        sigma_x=randint(1, sigma_x_max)
        sigma_y=randint(1, sigma_y_max)


        # RANDOM FIXATION POINTS
        center=[int(width/2.0), int(height/2.0)]

        # Convert np array to cv::Mat object
        my_mat_img = npcv.test_np_mat(img)

        # RANDOM FOVEA SIZE
        #my_lap_obj.update_fovea(width,height,sigma_x,sigma_y)

        # Foveate the image
        foveated_img = my_lap_obj.foveate(img,npcv.test_np_mat(np.array(center)))

        end = time.time()
        print(end - start)

        # Display the foveated image
        plt.imshow(foveated_img)
        #img.set_data(im)

        circle=plt.Circle((center[0],center[1]),1.0,color='blue')
        ax = plt.gca()
        ax.add_artist(circle)


        plt.draw()
        plt.pause(.001)
        plt.cla()



except KeyboardInterrupt:
    print('interrupted!')



