import SimpleITK as sitk
import sys
import os
import random

def loaddingList(relativeFolderPath,phase= None): 
    ''' 
    args : path to check, phase ('init' => check in the directory, else => check in ./images and ./labels of the directory)
    exit : dictionary{"flair":[list of flair path in dir],"seg":[list of labels path in dir]}
    '''

    absolute_path = os.path.dirname(__file__)
    dir_abs_path = os.path.join(absolute_path, relativeFolderPath)
    flair=[]
    seg=[]
    dic = {"flair":[],"seg":[]}

    if phase == 'init':
        pathList = os.listdir(dir_abs_path)
        for path in pathList:
            if (path!=('train'))&(path!=('val'))&(path!=('test')):
                if ((((path.rsplit('_', 1))[1]).rsplit('.',2))[0]) == 'flair':
                    flair.append(dir_abs_path+'/'+path)
                else :
                    seg.append(dir_abs_path+'/'+path)

        if len(flair) !=len(seg):
            print ("ERROR: not the same amout of segments and flaires")
            exit(1)
    
    else : 
        pathList = os.listdir(dir_abs_path+'/images')+os.listdir(dir_abs_path+'/labels')

        for path in pathList:
                if ((((path.rsplit('_', 1))[1]).rsplit('.',2))[0]) == 'flair':
                    flair.append(dir_abs_path+'/images/'+path)
                else :
                    seg.append(dir_abs_path+'/labels/'+path)

    flair.sort()
    seg.sort()
    dic['flair']=flair
    dic['seg']=seg

    return dic

def repartitor(dic,relative_path):
    ''' 
    args : dictionary{"flair":[list of flair path in dir],"seg":[list of labels path in dir]}, dir path contening 'test', 'val', 'train'
    exit : none
    aim : split and move dataset into train, test and val
    '''

    taille = len(dic['flair'])

    if taille!=0 :
        database = ['test','train','val']
        repartition = [0,int(taille*0.2),int(taille*0.2)+int(taille*0.8*0.8),taille]

        for i in range(len(repartition)-1):
            for j in range(repartition[i],repartition[i+1]):
                os.system('mv '+dic['flair'][j]+' '+relative_path+'/'+database[i]+'/images')
                os.system('mv '+dic['seg'][j]+' '+relative_path+'/'+database[i]+'/labels')


def reachSeuil(seuil,img,peak):
    ''' 
    args : seuil(int btwin 0 and 100), img(sitk.image), peak(int)
    exit : bool
    aim : says if (labeled pixels/pixels) >= seuil
    '''
    sumFilter = sitk.StatisticsImageFilter()
    x_slice = img.GetWidth()
    y_slice = img.GetHeight()
    nbPixels = int(x_slice*y_slice)
    sumFilter.Execute(img[:,:,peak])
    sge = sumFilter.GetSum()
    if (sge*100/nbPixels)>= seuil:
        return True
    return False

def seuilMax(img, nb_of_slice):
    ''' 
    args : nb_of_slice(int), img(sitk.image)
    exit : float
    aim : return the max seuil askable to have nb_of_slice minimum extract from the image
    '''

    sumFilter = sitk.StatisticsImageFilter()
    x_slice = img.GetWidth()
    y_slice = img.GetHeight()
    z_slice = img.GetDepth()
    nbPixels = int(x_slice*y_slice)
    data=[]
    max = -1
    if nb_of_slice>z_slice:
        print("ERROR: nb of silce > depth")
        exit(1)
    for z in range(z_slice):
        sumFilter.Execute(img[:,:,z])
        sge = sumFilter.GetSum()
        sgeProp = sge*100/nbPixels
        data.append(sgeProp)
    data.sort(reverse=True)
    max= data[int(seuil)]
    return max


def slicer_main(relativeFolderPath, seuil, testSeuil=False):

    path_img_dic=loaddingList(relativeFolderPath,'init')
    repartitor(path_img_dic,relativeFolderPath)

    testResult=[]
    database = ['test','val','train']

    for folder in database:
        path_img_dic = loaddingList(relativeFolderPath+"/"+folder)

        os.system("rm -r "+relativeFolderPath[: -4]+'/refined/'+folder+'/images/*')
        os.system("rm -r "+relativeFolderPath[: -4]+'/refined/'+folder+'/labels/*')

        for i in range(int(len(path_img_dic['flair']))):
            path = path_img_dic['seg'][i]
            pathFlair = path_img_dic['flair'][i]
            imgSeg = sitk.ReadImage(path)
            print(path)

            if testSeuil == True:
                    maxSeg = seuilMax(imgSeg,seuil)
                    testResult.append(maxSeg)
                    print("seg "+str(i)+" : seuil max ="+str(maxSeg)+" pour "+str(seuil)+" img/seg")

            else:
                castFilter = sitk.CastImageFilter()
                imgFlair = sitk.ReadImage(pathFlair)

                z_slice=imgSeg.GetDepth()
                nbOfSlice = 0
                size = list(imgSeg.GetSize())
                if len(size)==3:
                    size[2]=0
                Extractor = sitk.ExtractImageFilter()  
                peaklog = [i for i in range(z_slice)]

                while(nbOfSlice<int(testSeuil)) & (peaklog!=[]):
                    peak = random.randint(0,len(peaklog)-1)

                    if (reachSeuil(seuil, imgSeg, peaklog[peak])):
                        nbSlice=peaklog[peak]
                        strSegList=path.split('/')
                        newSegPath="/"
                        strFlairList=pathFlair.split('/')
                        newFlairPath="/"

                        for j in range(len(strSegList)):
                            if strSegList[j] == 'raw':
                                nPatient= strSegList[j+3].rsplit('_',4)[2]
                                newSegPath = str(newSegPath+'/refined/'+folder+"/labels/BRATS_"+nPatient+"_e"+str(nbSlice)+".png")
                                break
                            else:
                                newSegPath = str(newSegPath+"/"+strSegList[j])
                        
                        for j in range(len(strFlairList)):
                            if strFlairList[j] == 'raw':
                                nPatient= strFlairList[j+3].rsplit('_',4)[2]
                                newFlairPath = str(newFlairPath+'/refined/'+folder+"/images/BRATS_"+nPatient+"_e"+str(nbSlice)+".png")
                                break
                            else:
                                newFlairPath = str(newFlairPath+"/"+strFlairList[j])
                        
                        index = [0, 0, nbSlice]
                        Extractor.SetSize(size)
                        Extractor.SetIndex(index)

                        print(newSegPath)
                        print(newFlairPath)

                        castFilter.SetOutputPixelType(sitk.sitkUInt8)
                        imgFlair = sitk.RescaleIntensity(imgFlair,0,255)
                        imgFlairSmooth = castFilter.Execute(imgFlair)

                        castFilter.SetOutputPixelType(sitk.sitkUInt8)
                        imgSegSmooth = castFilter.Execute(imgSeg)

                        sitk.WriteImage(Extractor.Execute(imgSegSmooth), str(newSegPath),imageIO='PNGImageIO')
                        sitk.WriteImage(Extractor.Execute(imgFlairSmooth), str(newFlairPath),imageIO='PNGImageIO')

                        nbOfSlice = nbOfSlice +1

                        del(peaklog[peak])

                    else:
                        del(peaklog[peak])

        if testSeuil == True:
            testResult.sort()
            print("seuil max de "+folder+" pour "+str(seuil)+" img/seg = "+str(testResult[0])+"%")


if __name__ == "__main__":
    relative_path = sys.argv[1] 
    seuil = float(sys.argv[2])
    test = sys.argv[3]

    #__ATTENTION__: il faut que les fichiers compressés d'origines soit dans le dossier contenant les répertoires 'test', 'val' et 'train' raw
    #__ATTENTION__:les répertoires 'test', 'val' et 'train' doivent contenir un fichier 'images' et 'labels'

    #_Rq_: le script répartit directement les fichiers .nii entre train(80 * 80%), val(20 * 20%) et test(20%) du fichier raw 

    #executer le script dans le cli comme suit pour tester dataset: python3 slicer.py chemin_vers_repertoire_contenant_les_".nii" nb_clichés_par_patient True

    if sys.argv[3] == "True":
        print("#_____________TEST_____________#")
        slicer_main(relative_path, seuil,testSeuil=True)

    #executer le script dans le cli comme suit pour former dataset: python3 slicer.py chemin_vers_repertoire_contenant_fichier_'test','val'_et_'train'_(raw) seuil_tumeur/nb_pixel(entre 1 et 100) nb_clichés_par_patient_MAX
    else:
        print('#_____________SELECTION_____________#')
        slicer_main(relative_path, seuil, test)