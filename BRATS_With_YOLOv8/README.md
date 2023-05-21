Ce répertoire présente un implémentation de YOLOv8 ayant comme but la segmentation de tumeur type BRATS

Pour accéder au projet, allez dans ./code/Brats_with_Yolov8.ipynb.

Pour plus d'information sur les scripts utilisés, allez les voir directement. Chaque fonction est décrite avec les entrées, sorties et ce qu'elle fait.

__ATTENTION__ : créez bien les répertoires dataset comme indiqué dans ./code/Brats_with_Yolov8.ipynb

__ATTENTION__ : les codes et le notebook sont directement issus d'un environnement jupyterLab sour saturnCloud (image saturn-python), s'il y a des problèmes, il est très probable que cela vienne de l'environnement (sur saturn tout fonctionnaient bien).

__ATTENTION__ : la bibliothéque yolotools est trés à cheval sur les paths que l'on lui donne, faites bien attention à respecter l'archi du dataset pour éviter les erreurs.

__ATTENTION__ : ne prend à priori pas en charge la segmentation multi-objet dû a la fonction cv2.findContours qui renvoit des données en [[[int]],...] (pb facilement résolvable en y passant un peu de temps à mon avis).
