from rembg import remove
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

class ImageProcessor:
    def __init__(self,path,imageSize=512):
        self.path = path
        self.imageSize = imageSize
        self.originalImage = self.resizeImage(self.path, self.imageSize)
        self.image = self.originalImage.copy() 
        self.bglessImage = None
        self.alphaOnly = None
        self.Y = None
        self.Cb = None
        self.Cr = None

    #Resize image to specifed size for further processing
    @staticmethod
    def resizeImage(image_path,size):
        resize = transforms.Resize((size, size))
        imageInput = Image.open(image_path).convert("RGB")
        return resize(imageInput)        

    #Remove background of the input image
    def removeBackground(self):
        imageInput = self.originalImage
        imageOutput = remove(imageInput)
        self.bglessImage = imageOutput

    #Extract the transparency from the backgroundlessImage
    def extractAlpha(self):
        if self.bglessImage != None:
            rgbaImg = np.array(self.bglessImage.convert("RGBA")) #Convert to 4 channel image
            alpha = rgbaImg[:,:,3] # Extract alpha channel
            normalisedAlpha = alpha / 255.0 #normalise to 0-1
            self.alphaOnly = normalisedAlpha
        else:
            print("No backgroundless image available")

    #Display available image
    def displayImages(self):
        imgs = [self.originalImage, self.bglessImage, self.alphaOnly]
        titles = ["Original", "Background Removed", "Alpha Only"]
        validImg = [(img,title) for img, title in zip(imgs, titles) if img is not None]
        count = len(validImg)
        plt.figure(figsize=(count*4,count*2))
        for i,(img, title) in enumerate(validImg):
            plt.subplot(1,count,i+1)
            plt.imshow(img, cmap = "grey" if "Alpha Only"== title else None) #set grey cmap for alpha image
            plt.title(title)
            plt.axis("off")
        plt.show()

    def alphaOutput(self, inverse = False):
        alpha = self.alphaOnly
        if inverse :
            alpha = 1.0 - alpha
        return alpha
        
    def tensorisedImage(self, addDimension = 0):
        #Convert to tensor
        tensorise = transforms.ToTensor()
        tensorisedImage = tensorise(self.image)

        #Add dimension
        for i in range(addDimension):
            tensorisedImage = tensorisedImage.unsqueeze(0)
    
        return tensorisedImage

    def resizedImage(self):
        return self.originalImage

    def convertLuminanceChrominance(self):
        return self.originalImage.convert("YCbCr")

    def extractLuminanceDetail(self):
        luminanceImg = self.convertLuminanceChrominance()
        self.Y, self.Cb, self.Cr = luminanceImg.split()
        
        #Convert back to RGB with same Y value to keep shape
        self.replaceImage(Image.merge("RGB", (self.Y,self.Y,self.Y)))

    def applyChrominance(self, stylisedLuminance):
        #Check if Cb and Cr exist, else throw error
        if self.Cb is None or self.Cr is None:
            raise Exception("No Cb Cr")

        #Extract Y value from the Y-value only RGB image
        styledLuminance = stylisedLuminance.split()[0]

        #convert back to YCbCr and merge with extracted CbCr
        chromImg =  Image.merge("YCbCr", (styledLuminance, self.Cb, self.Cr)).convert("RGB")
        return np.array(chromImg).astype(np.float32) /255 #return as normlaised numpy

    def replaceImage(self, img):
        self.image = img

    @staticmethod
    def luminanceMatching(content, style):
        luminanceContent = np.array(content.split()[0])
        luminanceStyle = np.array(style.split()[0])

        stdContent = luminanceContent.std()
        meanContent = luminanceContent.mean()

        #Prevent divison by zero
        stdStyle = max(luminanceStyle.std(), 0.00001)
        meanStyle = luminanceStyle.mean()

        #Formular by Gatsy
        #Ls' = (STDc/STDs) * (Ls - MEANs) + MEANc
        matchedStyle = (stdContent/stdStyle) * (luminanceStyle - meanStyle) + meanContent
        matchedStyle = np.clip(matchedStyle, 0, 255) #Constrain

        #Convert to back to RGB image
        Y = Image.fromarray(matchedStyle, mode="L")
        return Image.merge("RGB", (Y, Y, Y))
        

            

    
