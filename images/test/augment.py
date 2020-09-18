import imgaug.augmenters as iaa
import cv2
import xml.etree.ElementTree as ET

filename = "skaiciaitest"
filename_with_extension = filename + ".xml"

height = 512
width = 512

label_org_xml = ET.parse(filename_with_extension)

annotation = label_org_xml.getroot()

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


label_org_xml.write(filename + "flipped.xml")

seq = iaa.Sequential([
    iaa.Fliplr(0.5),
])

imglist = []

img = cv2.imread(filename + ".png")

imglist.append(img)

images_aug = seq.augment_images(imglist)

cv2.imwrite(filename + "flipped.jpg", images_aug[0])