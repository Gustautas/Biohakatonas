import imgaug.augmenters as iaa
import cv2
import xml.etree.ElementTree as ET
import os

images_list = []

for file in os.listdir("./"):
    if file.endswith(".xml"):
        images_list.append(file)


for image in images_list:
    filename = image
    label_org_xml = ET.parse(filename)

    annotation = label_org_xml.getroot()

    for child in annotation.findall("object"):
        name = child.find("name")

        if(name.text) == "1/4":
            print("CHANGING")
            name.text = "0.25"
        elif(name.text) == "2/4":
            name.text = "0.5"
        elif(name.text) == "3/4":
            name.text = "0.75"
        elif(name.text) == "4/4":
            name.text = "1"

    label_org_xml.write(filename)