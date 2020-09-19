import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

images_list = []
for file in os.listdir("labeled_real_data_augmented_tik_simo"):
    if file.endswith(".jpg"):
        images_list.append(file)

for filename in images_list:
    img = cv2.imread('labeled_real_data_augmented_tik_simo/' + filename)
    dst = cv2.fastNlMeansDenoisingColored(img,None,50,21, 0,10)

    # fig = plt.figure()
    # fig.canvas.manager.full_screen_toggle()
    # plt.subplot(121),plt.imshow(img)
    # plt.subplot(122),plt.imshow(dst)
    # plt.show()

    cv2.imwrite("denoisedImages/" + filename, dst)