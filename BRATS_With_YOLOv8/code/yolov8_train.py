import sys
from ultralytics import YOLO


#BRATS
#model = YOLO('/home/jovyan/workspace/Yolov8/runs/detect/BRATS2/weights/best.pt'
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
    
#train BRATS detection
results = model.train(

   data=yaml,
   task=tache,
   imgsz=128,
   epochs=30,
   batch=10,
   name='BRATS_Seg',
   workers = 12,
)
