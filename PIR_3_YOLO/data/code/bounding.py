import numpy as np
import json
import os
import sys
import cv2 as cv
import glob
import yoloTools as yolo
from picsellia.types.enums import InferenceType

def polygonalisedContour(file_path):
        
        im = cv.imread(file_path)
        assert im is not None, "file could not be read, check with os.path.exists()"
        imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(imgray, 0, 255, 3)

        kernel = np.ones((5, 5), np.uint8)
        thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        #print(contours)

        for cnt in contours:
                approx = cv.approxPolyDP(cnt, 0.01*cv.arcLength(cnt, True), True)
                (x,y)=cnt[0,0]
        
        return (file_path,approx)



def trainDictionary(dico,file_path,contours):

        try : 
                id = len(dico["images"])
        except :
                id = 0

        im = cv.imread(file_path)
        height = im.shape[0]
        width = im.shape[1]
        poly  = []
        file =""

        for wrd in file_path.rsplit("/",12): 
                if wrd == "labels":
                        file = file+"/"+"images"
                else:
                        file = file+"/"+wrd

        dico["images"].append(  {"id": id,
                                "file_name":file,
                                "height": height,
                                "width": width
                                })
        
        for segment in contours :
                for point in segment:
                        for coord in point:
                                poly.append(str(coord))

        dico["annotations"].append(     {"id": id,
                                        "image_id":id,
                                        "segmentation": poly,
                                        "category_id": 1,
                                        "iscrowd": 0
                                        })
        return dico


def convertToCOCO(dico):
        json_Path = "declaration_labels.json"
        with open(json_Path,"w") as f:
                json_string = json.dump(dico,f)
                return json_Path

def convertToYolo(COCO_Path, img_Path):
        converter = yolo.YOLOFormatter(
                fpath = COCO_Path,
                imdir = img_Path,
                #steps = ["train"],
                mode = InferenceType.SEGMENTATION
        )

        converter.convert()
        yaml_fp = converter.generate_yaml(dpath = os.path.join(img_Path,'data.yaml'))

        
if __name__ == "__main__":

        try:
                dossier=sys.argv[1]
        except:
                print("ERROR: write the file's relative path")
                exit(1)

        absolute_path = os.path.dirname(__file__)
        dir_abs_path = os.path.join(absolute_path, dossier)
        file_path = []

        for path in os.listdir(dir_abs_path):
                file_path.append(os.path.join(dir_abs_path, path))

        #print (file_path)

        dico = {"info":{},"licenses":[],"images":[],"annotations":[],"categories":[{"id":1,"name":"tumeur"}]}

        for img in file_path:

        #img = "/home/amfarwati/Documents/PIR_3/data/BRATS/refined/train/labels/BRATS_001_e70.png"
                file, contour = polygonalisedContour(img)
                dico = trainDictionary(dico,file,contour)

        jsonCOCO_Path = convertToCOCO(dico)
        convertToYolo(jsonCOCO_Path, dossier.rsplit('/', 2)[0]+'/')
