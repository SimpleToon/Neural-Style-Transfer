class ImageProcessor:
    def __init__(self,path,imageSize=512):
        self.path = path
        self.imageSize = imageSize
        self.originalImage = self.resizeImage(self.path, self.imageSize)
        self.bglessImage = None
        self.alphaOnly = None

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
        tensorisedImage = tensorise(self.originalImage)

        #Add dimension
        for i in range(addDimension):
            tensorisedImage.unsqueeze(0)
    
        return tensorisedImage

    def resizedImage(self):
        return self.originalImage
