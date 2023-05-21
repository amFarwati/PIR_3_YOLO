import numpy as np
import json
import os
import sys
import cv2 as cv
import yoloTools
from picsellia.types.enums import InferenceType

def rescale(file_path, sqrDim, inter_type):
    ''' 
    args : file_path of the image,sqrDim (int), inter_type (iterpolation parameter of cv2)
    exit : none
    aim : resize the image defined by file_path as a sqrDim side square image
    '''

    print(file_path)
    img = cv.imread(file_path)
    if (img.shape[0]!=img.shape[1])|(img.shape[0]!=sqrDim):
        w = sqrDim
        h = sqrDim
        dim = (w,h)
        img = cv.resize(img, dim, interpolation = inter_type)
        cv.imwrite(file_path, img)

def polygonalisedContour(file_path,type_analyse):
        ''' 
        args : file_path of the label image,type_analyse ('bbox' or 'seg')
        exit : file_path of the label image (str), approx [[[int]],[[int],...]] (list of the polygon coordonnates)
        aim : resize the image defined by file_path as a sqrDim side square image
        '''
        im = cv.imread(file_path)
        assert im is not None, "file could not be read, check with os.path.exists()"
        imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(imgray, 0, 255, 3)
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
                approx = cv.approxPolyDP(cnt, 0.01*cv.arcLength(cnt, True), True)
                (x,y)=cnt[0,0]

        if type_analyse == 'bbox':
            x,y,w,h = cv.boundingRect(approx)
            approx[0] = int(x+w/2)
            approx[1] = int(y+h/2)
            approx[2] = w
            approx[3] = h
        return (file_path,approx)

def trainDictionary(dico,file_path,contours,type_analyse):
        ''' 
        args : dico = {"info":{},"licenses":[],"images":[],"annotations":[],"categories":[{"id":0,"name":"tumeur"}]}, file_path of the label image, contours [[[int]],...], type_analyse ('bbox' or 'seg')
        exit : dico filled with image and label data refered to file_path
        aim : fill the dico with image and label data as a COCO format
        ''' 
        try : 
                id = len(dico["images"])
        except :
                id = 0
        im = cv.imread(file_path)
        height = im.shape[0]
        width = im.shape[1]
        poly  = []
        file =""
        labelFile = 'labels'
        for wrd in file_path.rsplit("/",12): 
                if wrd == labelFile:
                        file = file+"/"+"images"
                else:
                        file = file+"/"+wrd
        dico["images"].append(  {"id": id,
                                "file_name":file,
                                "height": height,
                                "width": width
                                })
        if type_analyse == 'seg':
                for seg in contours:
                        for point in seg:
                                for coord in point:
                                        poly.append(str(coord))  

        elif type_analyse == 'bbox':
            
            for coord in contours:
                poly.append(str(coord))
                
        dico["annotations"].append(     {"id": id,
                                        "image_id":id,
                                        "segmentation": poly,
                                        "category_id": 0,
                                        "iscrowd": 0
                                        })
                
        os.system("rm -r "+file_path)

        return dico

def convertToCOCO(dico):
        ''' 
        args : dico = {"info":{},"licenses":[],"images":[],"annotations":[],"categories":[{"id":0,"name":"tumeur"}]}
        exit : json_Path to the json COCO
        aim : create json form the dictionary as a COCO format
        ''' 
        json_Path = "declaration_labels.json"
        with open(json_Path,"w") as f:
                json_string = json.dump(dico,f)
                return json_Path

def convertToYolo(COCO_Path, img_Path):
        ''' 
        args : COCO_Path to the json file, img_Path to the directory containing test, val and train directories 
        exit : none
        aim : générate data.yaml and labels.txt decription in yolo format + delete all the label images 
        ''' 
        converter = yoloTools.YOLOFormatter(
                fpath = COCO_Path,
                imdir = img_Path,
                mode = InferenceType.SEGMENTATION
        )
        converter.convert()
        yaml_fp = converter.generate_yaml(dpath = os.path.join(img_Path,'data.yaml'))
        
def formatage_main(refined_path, type_analyse, taille_carrée):
    
    database = ["/train/labels","/val/labels","/test/labels"]

    refined_path = os.path.join(os.path.dirname(__file__),refined_path)
    sqrSize = int(taille_carrée)
    file_path = []
    labelFile = 'labels'

    for path in database:
        command  = "rm -r "+refined_path+path+"/*.txt"
        os.system(command)

    for path in os.listdir(refined_path+database[0]):
        if not path.startswith('.'):
            file_path.append(os.path.join(refined_path+database[0], path))

    for path in os.listdir(refined_path+database[1]):
        if not path.startswith('.'):
            file_path.append(os.path.join(refined_path+database[1], path))

    for path in os.listdir(refined_path+database[2]):
        if not path.startswith('.'):
            file_path.append(os.path.join(refined_path+database[2], path))

    dico = {"info":{},"licenses":[],"images":[],"annotations":[],"categories":[{"id":0,"name":"tumeur"}]}

    for img in file_path:
        rescale(img,sqrSize,cv.INTER_LINEAR)
        img_flair= ""
        for x in img.rsplit("/"):
            if x == labelFile:
                img_flair = img_flair + "/images"
            else: 
                img_flair = img_flair +'/'+ x 

        print (img)        
        print (img_flair)

        rescale(img_flair,sqrSize,cv.INTER_LINEAR)
        file, contour = polygonalisedContour(img,type_analyse)
        dico = trainDictionary(dico,file,contour,type_analyse)

    jsonCOCO_Path = convertToCOCO(dico)

    convertPath = ''
    for wrd in refined_path.rsplit('/'):
        if (wrd=='test')|(wrd=='train')|(wrd=='val'):
              break
        elif (wrd == 'code')|(wrd=='..'):
              pass
        else:
              convertPath= convertPath+wrd+'/'
              

    print("convertpath "+convertPath)
    convertToYolo(jsonCOCO_Path, convertPath)
        
if __name__ == "__main__":

        #__ATTENTION__:les répertoires 'test', 'val' et 'train' doivent contenir un fichier 'images' et 'labels'
        #executer le script dans le cli comme suit : python3 bounding.py chemin_vers_repertoire_contenant_les_"train,test,val" mode(seg ou bbox) resolution_resize(coté carré)

        #Rq le fichier data.yaml sera générer dans le fichier contenant les repertoires 'test', 'val' et 'train'
        #Rq le fichier delaration_labels.json (dataset au format COCO) sera générer dans le fichier contenant les repertoires le script bounding.py

        try:
                dossier=sys.argv[1]
        except:
                print("ERROR: write the file's relative path")
                exit(1)
        absolute_path = os.path.dirname(__file__)
        dir_abs_path = os.path.join(absolute_path, dossier)

        formatage_main(dir_abs_path,sys.argv[2],sys.argv[3])