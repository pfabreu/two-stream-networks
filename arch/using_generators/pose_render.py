import glob
import os
import json
import cv2
import numpy as np

_JOINT_DIR = "/media/pedro/actv3/AHA/pose_2d/joints/"
_VID_DIR = "/media/pedro/actv3/AHA/videos/original/"
_OUT_DIR = "/media/pedro/actv3/AHA/pose_facehands/"
generate_vids = True  # Set this flag to True if you want to also generate videos besides images

for vid_folder in glob.glob(_JOINT_DIR + "*"):
    print vid_folder
    vidname = vid_folder.rsplit('/', 1)[1]
    print vidname
    if os.path.exists(_VID_DIR + vidname + ".avi"):
        in_vid = cv2.VideoCapture(_VID_DIR + vidname + ".avi")
        width = int(in_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(in_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if generate_vids:
            out_face = cv2.VideoWriter(_OUT_DIR + vidname + "_face.avi", cv2.VideoWriter_fourcc(*"XVID"), 30, (width, height))
            out_hands = cv2.VideoWriter(_OUT_DIR + vidname + "_hands.avi", cv2.VideoWriter_fourcc(*"XVID"), 30, (width, height))
        for f in glob.glob(vid_folder + "/*.json"):
            # Go through all json files that represent
            with open(f) as json_f:
                # print f
                # This is a single frame
                data = json.load(json_f)
                blank_image = np.zeros((height, width, 3), np.uint8)
                blank_image_hands = np.zeros((height, width, 3), np.uint8)
                # pprint.pprint(data)
                # Create empy numpy array for this frame which will have size #nppl x #joints
                frame_poses = []
                # With data, you can now also find values like so:
                ppl = data["people"]

                p_count = 0
                for person in ppl:
                    # Render faces
                    jts = person['face_keypoints_2d']
                    jts = [jts[n:n + 3] for n in range(0, len(jts), 3)]

                    pixel_count = 0
                    for pixel in jts:
                        if pixel != [0, 0, 0]:
                            y_c = int(pixel[1])
                            x_c = int(pixel[0])
                            radius = int(0.0031 * width)
                            if x_c > radius and x_c < width - radius and y_c > radius and y_c < height - radius:
                                for y in range(y_c - radius, y_c + radius):
                                    for x in range(x_c - radius, x_c + radius):
                                        # If eyes
                                        if (pixel_count >= 36 and pixel_count <= 41 or pixel_count == 68) or (pixel_count >= 42 and pixel_count <= 47 or pixel_count == 69):
                                            blank_image[y, x] = (255, 0, 0)
                                        # If mouth
                                        elif pixel_count >= 48 and pixel_count <= 67:
                                            blank_image[y, x] = (0, 255, 0)
                                        else:
                                            # If face bones
                                            blank_image[y, x] = (0, 0, 255)
                            pixel_count += 1

                    jts = person['hand_left_keypoints_2d']
                    jts = [jts[n:n + 3] for n in range(0, len(jts), 3)]
                    pixel_count = 0
                    for pixel in jts:
                        if pixel != [0, 0, 0]:
                            y_c = int(pixel[1])
                            x_c = int(pixel[0])
                            radius = int(0.0031 * width)
                            if x_c > radius and x_c < width - radius and y_c > radius and y_c < height - radius:
                                for y in range(y_c - radius, y_c + radius):
                                    for x in range(x_c - radius, x_c + radius):
                                        # Thumb is red
                                        if pixel_count >= 0 and pixel_count <= 4:
                                            blank_image_hands[y, x] = (0, 0, 255)
                                        # Index is yellow
                                        elif pixel_count >= 5 and pixel_count <= 8:
                                            blank_image_hands[y, x] = (255, 255, 0)
                                        # Middle finger is green
                                        elif pixel_count >= 9 and pixel_count <= 12:
                                            blank_image_hands[y, x] = (0, 255, 0)
                                        # Anelar is blue
                                        elif pixel_count >= 13 and pixel_count <= 16:
                                            blank_image_hands[y, x] = (255, 0, 0)
                                        # Pinky is purple
                                        elif pixel_count >= 17 and pixel_count <= 20:
                                            blank_image_hands[y, x] = (255, 0, 255)
                        pixel_count += 1

                    jts = person['hand_right_keypoints_2d']
                    jts = [jts[n:n + 3] for n in range(0, len(jts), 3)]
                    pixel_count = 0
                    for pixel in jts:
                        if pixel != [0, 0, 0]:
                            y_c = int(pixel[1])
                            x_c = int(pixel[0])
                            radius = int(0.0031 * width)
                            if x_c > radius and x_c < width - radius and y_c > radius and y_c < height - radius:
                                for y in range(y_c - radius, y_c + radius):
                                    for x in range(x_c - radius, x_c + radius):
                                        # Thumb is red
                                        if pixel_count >= 0 and pixel_count <= 4:
                                            blank_image_hands[y, x] = (0, 0, 255)
                                        # Index is yellow
                                        elif pixel_count >= 5 and pixel_count <= 8:
                                            blank_image_hands[y, x] = (255, 255, 0)
                                        # Middle finger is green
                                        elif pixel_count >= 9 and pixel_count <= 12:
                                            blank_image_hands[y, x] = (0, 255, 0)
                                        # Anelar is blue
                                        elif pixel_count >= 13 and pixel_count <= 16:
                                            blank_image_hands[y, x] = (255, 0, 0)
                                        # Pinky is purple
                                        elif pixel_count >= 17 and pixel_count <= 20:
                                            blank_image_hands[y, x] = (255, 0, 255)
                        pixel_count += 1
                out_face.write(blank_image)
                out_hands.write(blank_image_hands)
        cv2.destroyAllWindows()
        out_face.release()
        out_hands.release()
print "All pose segments processed!"
