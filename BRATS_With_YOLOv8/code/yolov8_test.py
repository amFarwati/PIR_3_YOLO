import yoloTools
import os
import sys
from picsellia.types.enums import AnnotationFileType
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import glob
import torch


#BRATS

type_reseau = sys.argv[1]
tache = ' '

if type_reseau == 'bbox':
    model = YOLO('yolov8l.yaml')
    yaml = 'dataBBOX.yaml'
    tache = 'detect'
    
elif type_reseau == 'seg':
    model = YOLO('yolov8l-seg.yaml')
    yaml = 'dataSEG.yaml'
    tache = 'segment'
    
#predict BRATS
results = model.predict(source = yaml,
                         save=True,
                         show=True,
                         imgsz=128,
                         conf=0.5)

