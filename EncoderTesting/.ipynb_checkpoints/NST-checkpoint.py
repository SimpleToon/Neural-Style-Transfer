from rembg import remove
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from ImageProcessor import ImageProcessor
from LPIPS import LPIPS
import time


class NST:
    def __init__(self, prebuild_encoder = None, prebuild_decoder = None, device=None):
        self.content_path = None
        self.style_paths = ()
        self.device = device or torch.device("cpu")
        self.encoder = None
        self.decoder = None
        self.prebuild_encoder = prebuild_encoder
        self.prebuild_decoder = prebuild_decoder
        self.content = None
        self.styles = ()
        self.stylisedImage = None
        self.stylisedTensor = None
        self.start = None
        self.end = None
        self.tensorisedContent = None
        self.tensorisedStyles = None

    def saveImage(self, name="Image"):
        plt.imsave(f"output/{name}.jpg", self.stylisedImage)

    def evaluate(self):
        lpips = LPIPS()
        score = lpips.eval(self.content.tensorisedImage().cpu(), self.stylisedTensor.cpu())

        #Evaluate time passed
        if self.start is not None and self.end is not None:
            #in millisecond
            timePassed = (self.end - self.start) * 1000
        else: 
            timePassed = None
            
        return round(score,3), f"{timePassed:.3f} ms"

    def reset(self):
        self.tensorisedContent = None
        self.tensorisedStyles = None
        self.content = None
        self.styles = ()
        self.stylisedImage = None
        self.stylisedTensor = None
        self.content_path = None
        self.style_paths = ()


    def fit(self, content_path, styles_path):
        self.reset()
        self.content_path = content_path
        self.style_paths = styles_path
        self.processor(1)

    #Pre-processing of image
    def processor(self, addDimension = 0):
        #Load content
        self.content = ImageProcessor(self.content_path)

        #Clear stored styles to avoid duplication
        self.styles = ()
        
        #Load styles
        for style in self.style_paths:
            self.styles += (ImageProcessor(style),)

        content = self.content
        styles = self.styles

        #Convert to tensor
        self.tensorisedContent = content.tensorisedImage(addDimension).to(self.device)
        self.tensorisedStyles = tuple( s.tensorisedImage(addDimension).to(self.device) for s in styles)



    def encodeAll (self, content, styles):
        #Enocde content and styles
        contentFeature = self.encoder(content)
        styleFeatures = tuple(self.encoder(s) for s in styles)
        return contentFeature, styleFeatures


    def decoding(self, output):
        #Decode output
        decodedOutput = self.decoder(output)
        #Convert back to image - remove tensor, convert to numpy, and restructure
        self.stylisedTensor = decodedOutput.detach().clamp(0, 1)
        self.stylisedImage = self.stylisedTensor.squeeze(0).permute(1,2,0).cpu().numpy()
        return self.stylisedImage
        

    #Display available image
    def displayImages(self):
        content = self.content.resizedImage()
        styles = [s.resizedImage() for s in self.styles]
        stylised = self.stylisedImage

        #Create title 
        titles = (["Content"]+ [f"Style {i+1}" for i in range(len(styles))] + ["Stylised"])

        #Combine as list
        imgs = [content] + styles + [stylised] 

        #filter out None image
        validImg = [(img,title) for img, title in zip(imgs, titles) if img is not None]

        count = len(validImg)

        #Display image
        plt.figure(figsize=(count*4,count*2))
        for i,(img, title) in enumerate(validImg):
            plt.subplot(1,count,i+1)
            plt.imshow(img)
            plt.title(title)
            plt.axis("off")
        plt.show()

    #Call pipeline and test time
    def pipeline(self, callback, addDimension = 1):
        self.start = time.perf_counter()
        #Encode image
        n_c,n_s = self.encodeAll(self.tensorisedContent,self.tensorisedStyles)
        #Transformer
        features = callback(n_c,n_s)
        #Decode
        self.decoding(features)
        self.end = time.perf_counter()
        
    #Load prebuild model
    def loadPrebuildEncoder(self, layers = None):
        eState = torch.load(self.prebuild_encoder, map_location="cpu")
        self.encoder.load_state_dict(eState)
        if layers != None: #slice layer if applied
            self.encoder = nn.Sequential(*list(self.encoder.children())[:layers])
            self.encoder = self.encoder.to(self.device)
        
    def loadPrebuildDecoder(self, layers = None):
        eState = torch.load(self.prebuild_decoder, map_location="cpu")
        self.decoder.load_state_dict(eState)
        if layers != None:
            self.decoder = nn.Sequential(*list(self.decoder.children())[:layers])
            self.decoder = self.decoder.to(self.device)
    
    #helper tool to evaluate encoder/decoder
    def evalEncoder(self):
        self.encoder.eval()
        
    def evalDecoder(self):
        self.decoder.eval()

    #helper tool to unload non-default encoder/decoder
    def uploadEncoder(self, input):
        self.encoder = input
        self.encoder.to(self.device)
        
    def uploadDecoder(self, input):
        self.decoder = input
        self.decoder.to(self.device)