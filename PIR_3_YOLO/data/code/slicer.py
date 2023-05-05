import SimpleITK as sitk
import sys
import os
import random
import time


def loaddingList(relativeFolderPath):

    absolute_path = os.path.dirname(__file__)
    dir_abs_path = os.path.join(absolute_path, relativeFolderPath)
    print(os.path.join(absolute_path, relativeFolderPath))
    list=[]
    flair=[]
    seg=[]
    dic = {"flair":[],
           "seg":[]
           }
    
    pathList = os.listdir(dir_abs_path)

    for path in pathList:
        if ((((path.rsplit('_', 1))[1]).rsplit('.',2))[0]) == 'flair':
            flair.append(dir_abs_path+'/'+path)
        else :
            seg.append(dir_abs_path+'/'+path)

    if len(flair) !=len(seg):
        print ("ERROR: not the same amout of segments and flaires")
        exit(1)

    if len(flair)==0:
        print("No files found")
        exit(1)

    flair.sort()
    seg.sort()
    dic['flair']=flair
    dic['seg']=seg

    return dic

def reachSeuil(seuil,img,peak):

    #pixelVal = []
    x_slice = img.GetWidth()
    y_slice = img.GetHeight()
    z_slice = img.GetDepth()
    nbPixels = int(img.GetNumberOfPixels()/z_slice)
    nbSegs = 0

    for i in range(x_slice):
        for j in range(y_slice):
            #pixelVal.append(img.GetPixel(i,j,peak))
            if img.GetPixel(i,j,peak)!=0:
                #print(img.GetPixel(i,j,peak))
                nbSegs = nbSegs + 1

        if (nbSegs*100/nbPixels)>= seuil:
            return True
    #print(pixelVal)

    return False

def seuilMax(img, seuil):

    x_slice = img.GetWidth()
    y_slice = img.GetHeight()
    z_slice = img.GetDepth()
    nbPixels = int(img.GetNumberOfPixels()/z_slice)

    print("Nb Pixels stik fct:"+str(nbPixels))
    print("Nb Pixels man calcul:"+str(x_slice*y_slice))


    data=[]
    max = -1

    if seuil>z_slice:
        print("ERROR: nb of silce > depth")
        exit(1)

    for z in range(z_slice):
        sgeProp = -1
        sge = 0
        for x in range(x_slice):
            for y in range(y_slice):
                if img.GetPixel(x,y,z)> 0:
                    sge = sge +1
        sgeProp = sge*100/nbPixels
        data.append(sgeProp)
    data.sort(reverse=True)
    max= data[seuil]

    return max


def imgRoutine(relativeFolderPath, seuil, testSeuil=False):

    path_img_dic=loaddingList(relativeFolderPath)

    nbOfFiles = 0
    testResult=[]
    timer = time.monotonic()

    for i in range(int(len(path_img_dic['flair'])/3)):

        nbOfFiles = i + 1
        path = path_img_dic['seg'][i]
        pathFlair = path_img_dic['flair'][i]
        imgSeg = sitk.ReadImage(path) # some 3D volume

        if testSeuil == True:
                
                maxSeg = seuilMax(imgSeg,seuil)
                testResult.append(maxSeg)
                timer = time.monotonic()-timer
                print("seg "+str(i)+" : seuil max ="+str(maxSeg)+" pour "+str(seuil)+" img/seg")

        else:
            castFilter = sitk.CastImageFilter()
            imgFlair = sitk.ReadImage(pathFlair)

            z_slice=imgSeg.GetDepth()
            print(z_slice)
            nbOfSlice = 0
            size = list(imgSeg.GetSize())
            size[2]=0
            print(size)
            Extractor = sitk.ExtractImageFilter()  
            peaklog = [i for i in range(z_slice)]

            while(nbOfSlice<int(testSeuil)) & (peaklog!=[]):
                peak = random.randint(0,len(peaklog)-1)

                #print(peaklog)

                if (reachSeuil(seuil, imgSeg, peaklog[peak])):
                    
                    nbSlice=peaklog[peak]
                    strSegList=path.split('/')
                    newSegPath="/"
                    strFlairList=pathFlair.split('/')
                    newFlairPath="/"

                    for j in range(len(strSegList)):
                        if strSegList[j] == 'raw':
                            nPatient= strSegList[j+1].rsplit('_',4)[2]
                            newSegPath = str(newSegPath+'/refined/labels/'+"BRATS_"+nPatient+"_e"+str(nbSlice)+".png")
                            break
                        else:
                            newSegPath = str(newSegPath+"/"+strSegList[j])
                    
                    for j in range(len(strFlairList)):
                        if strFlairList[j] == 'raw':
                            nPatient= strFlairList[j+1].rsplit('_',4)[2]
                            newFlairPath = str(newFlairPath+'/refined/images/'+"BRATS_"+nPatient+"_e"+str(nbSlice)+".png")
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
                    print("seg "+str(i)+" OK in "+str(time.monotonic()-timer)+" s")

                else:
                    del(peaklog[peak])


            
    if testSeuil == True:
        testResult.sort()
        print("seuil max pour "+str(seuil)+" img/seg = "+str(testResult[0])+"%")

    print("files processed : "+str(nbOfFiles)+" in "+str(time.monotonic()-timer)+"s") 

#----------------------------------------------------------------

relative_path = sys.argv[1] 
seuil = float(sys.argv[2])
test = sys.argv[3]

if sys.argv[3] == "True":
    print("#_____________TEST_____________#")
    imgRoutine(relative_path, seuil,testSeuil=True)
else:
    print('#_____________SELECTION_____________#')
    imgRoutine(relative_path, seuil, test)
