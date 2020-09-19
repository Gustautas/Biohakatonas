import imgaug.augmenters as iaa
import cv2
import xml.etree.ElementTree as ET
import os

images_list = []
height = 512
width = 512

for file in os.listdir("labeled_real_data_non_augmented_tik_simo"):
    if file.endswith(".jpg"):
        images_list.append(file[:-4])


for image in images_list:
    filename = image
    filename_with_extension = filename + ".xml"


    label_org_xml = ET.parse("labeled_real_data_non_augmented_tik_simo/" + filename_with_extension)

    annotation = label_org_xml.getroot()

    annotation.find("filename").text = annotation.find("filename").text[:-4] + "flipped.jpg"
    annotation.find("path").text = annotation.find("path").text[:-4] + "flipped.jpg"

    for child in annotation.findall("object"):
        bndbox = child.find("bndbox")

        xmin = bndbox.find("xmin")
        xmin_text_old = xmin.text

        xmax = bndbox.find("xmax")
        xmax_text_old = xmax.text

        ymin = bndbox.find("ymin")
        ymin_text_old = ymin.text

        ymax = bndbox.find("ymax")
        ymax_text_old = ymax.text


        xmin.text = str(width - int(xmax_text_old))

        xmax.text = str(width - int(xmin_text_old))

        ymin.text = ymin_text_old

        ymax.text = ymax_text_old


    label_org_xml.write("labeled_real_data_augmented_tik_simo/" + filename + "flipped.xml")

    seq = iaa.Sequential([
        iaa.Fliplr(1),
    ])

    imglist = []

    img = cv2.imread("labeled_real_data_non_augmented_tik_simo/" + filename + ".jpg")

    imglist.append(img)

    images_aug = seq.augment_images(imglist)

    cv2.imwrite("labeled_real_data_augmented_tik_simo/" + filename + "flipped.jpg", images_aug[0])