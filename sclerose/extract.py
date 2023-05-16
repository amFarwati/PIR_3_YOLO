#
# This script lets you extract slices (with an N4 biais field correction) from .nii files
# Each slice is selected according to its segmentation rate (aka tolerance)
#

import sys, os, re, random
import SimpleITK as sitk
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

path = "MICCAI" # .nii files would be inside the "MICCAI" folder
flair = "_3DFLAIR.nii" # flair files suffix
seg = "_Consensus.nii" # seg files suffix
nbSlicesPerImage = 60

openingFilter = sitk.BinaryMorphologicalOpeningImageFilter()
openingFilter.SetKernelRadius(4)
openingFilter.SetKernelType(sitk.sitkBall)

corrector = sitk.N4BiasFieldCorrectionImageFilter()
corrector.SetMaximumNumberOfIterations([20]*4)

caster = sitk.CastImageFilter()
caster.SetOutputPixelType(sitk.sitkUInt8)


def getSlices(segPath):
    # This is the main function executed for each .nii seg file
    segImage = sitk.ReadImage(segPath, sitk.sitkUInt8)
    flairPath = segPath.replace(seg,flair)
    flairImage = sitk.ReadImage(flairPath)
    
    size = segImage.GetSize()
    totalPixel = size[0] * size[1]
    
    # N4 filtering
    maskImage = sitk.OtsuThreshold(flairImage, 0, 1,200)
    print(f"N4 filtering {fileName} ...")
    openedMaskImage = openingFilter.Execute(maskImage)

    #component_image = sitk.ConnectedComponent(openedMaskImage)
    #sorted_component_image = sitk.RelabelComponent(component_image, sortByObjectSize=True)
    #largest_component_binary_image = sorted_component_image == 1
    #sitk.WriteImage(largest_component_binary_image, "toto.nii.gz")

    #corrected_image = corrector.Execute(flairImage, largest_component_binary_image)
    corrected_image = corrector.Execute(flairImage, openedMaskImage)
    #corrected_image = corrector.Execute(flairImage)

    
    notTestedSlices = list(range(size[2]))
    savedSlices = []
    p_bar = tqdm(range(size[2]), desc=f"Extracting from {segPath}", unit="slices", file=sys.stdout)
    
    tolerance = 0.125 # initial segmentation rate for an image
    
    while len(savedSlices) < nbSlicesPerImage:

        if len(notTestedSlices) == 0: # Decrease tolerance if it was too high
            tolerance = tolerance/2
            if tolerance < 0.0001:
                print("   Tolerance is too small, stopping")
                break
            print (f"   Tolerance is now {tolerance}%")
            p_bar.close()
            p_bar = tqdm(range(size[2]), desc=f"Extracting from {segPath}", unit="slices")
            p_bar.refresh()
            notTestedSlices = list(range(size[2]))
            for sliceInd in savedSlices:
                notTestedSlices.remove(sliceInd)
                p_bar.update(1)
                p_bar.refresh()

        sliceInd = random.choice(notTestedSlices)
        notTestedSlices.remove(sliceInd)
        p_bar.update(1)
        p_bar.refresh()
        sliceArr = getArr(segImage, sliceInd)
        nonZeroRate = np.count_nonzero(sliceArr) * 100 / totalPixel
        if nonZeroRate < tolerance:
            continue
        else:
            savedSlices.append(sliceInd)
            seg2D = segImage[:, :, sliceInd]
            saveSlice(seg2D, sliceInd,segPath) # save seg slice
            flair2D = corrected_image[:, :, sliceInd]
            saveSlice(flair2D, sliceInd,flairPath) # save flair slice

    p_bar.close()


def getArr(image, sliceIndex):
    # Transform an image to a numpy array
    size = image.GetSize()
    sliceArr = np.zeros((size[1],size[0]))
    for x in range(size[0]):
        for y in range(size[1]):
            sliceArr[y][x] = image[x,y,sliceIndex]
    return sliceArr

def saveSlice(img2D, sliceIndex, segPath):
    # This function saves slices (FLAIR + SEG !) inside a subdirectory named "extracted"
    isSeg = segPath.endswith(seg)
    if not isSeg:
        img2D = sitk.RescaleIntensity(img2D, 0, 255)
    img2D = caster.Execute(img2D)

    match = re.search("(?<=\/)(.*?)(?=\.)",segPath)
    imName = segPath[match.start():match.end()]
    if imName.endswith("_3DFLAIR"):
        imName = imName.replace("_3DFLAIR","")
    elif imName.endswith("_Consensus"):
        imName = imName.replace("_Consensus","")

    match = re.search("(.*?)(?=\/)",segPath)
    folderName = segPath[match.start():match.end()]

    if not os.path.exists(f"{folderName}/extracted"):
        os.makedirs(f"{folderName}/extracted")
    if not os.path.exists(f"{folderName}/extracted/seg"):
        os.makedirs(f"{folderName}/extracted/seg")
    if not os.path.exists(f"{folderName}/extracted/flair"):
        os.makedirs(f"{folderName}/extracted/flair")
    
    if isSeg:
        sitk.WriteImage(img2D, f"{folderName}/extracted/seg/{imName}_{sliceIndex}.png")
    else:
        sitk.WriteImage(img2D, f"{folderName}/extracted/flair/{imName}_{sliceIndex}.png")
        
        
        
        
for fileName in os.listdir(path):
    if fileName.endswith(seg) and fileName.startswith(str(sys.argv[1])):
        getSlices(path + "/" + fileName)
