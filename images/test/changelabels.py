import imgaug.augmenters as iaa
import cv2
import xml.etree.ElementTree as ET
import os

label_org_xml = ET.parse("clusters1.xml")

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


    xmin.text = str(int(int(xmin_text_old) / 2))

    xmax.text = str(int(int(xmax_text_old) / 2))

    ymin.text = str(int(int(ymin_text_old) / 2))

    ymax.text = str(int(int(ymax_text_old) / 2))


label_org_xml.write("clusters1.xml")