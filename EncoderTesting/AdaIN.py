import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from NST import NST
from itertools import cycle
import time

#Use silicon gpu/cuda gpu if available
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


#implementation based on https://github.com/naoto0804/pytorch-AdaIN
class AdaIN(NST):
    def __init__(self, prebuild_encoder = None, prebuild_decoder = None, device = device):
        super().__init__(prebuild_encoder, prebuild_decoder, device)
        self.encoder = self._vgg()
        self.decoder = self.vggDecoder()
        self.mse_loss = nn.MSELoss()
        
    def setup(self, encoderSlice = 31, decoderSlice = None):
        if self.prebuild_encoder is not None:
            self.loadPrebuildEncoder(encoderSlice) 
        if self.prebuild_decoder is not None :
            self.loadPrebuildDecoder(decoderSlice) 
        self.encoder.to(self.device).eval()
        self.decoder.to(self.device)
        

    #Encoder by nato0804
    def _vgg(self):
        return nn.Sequential(
            nn.Conv2d(3, 3, (1, 1)),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 64, (3, 3)),
            nn.ReLU(),  
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),  
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),  
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),  
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(),  
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU(),  # relu4-1, this is the last layer used
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(), 
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU()  
        )

    #Decoder by nato0804
    def vggDecoder(self,  inputSize = 512):
        return nn.Sequential(    
            nn.ReflectionPad2d(1),
            nn.Conv2d(inputSize, 256, 3),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, 3),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(inplace=True),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, 3),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, 3),
        )

    #Adain implementation by nato0804
    #Calcualte mean and variance
    def _calc_mean_std(self, feat, eps=1e-5):
        N, C = feat.size()[:2]
        feat_ = feat.view(N, C, -1)
        mean = feat_.mean(dim=2).view(N, C, 1, 1)
        std = feat_.var(dim=2).add(eps).sqrt().view(N, C, 1, 1)
        return mean, std

    #Matching style with features 
    def _adain(self, content_feat, style_feat, alpha=1.0):
        c_mean, c_std = self._calc_mean_std(content_feat)
        s_mean, s_std = self._calc_mean_std(style_feat)
        normalized = (content_feat - c_mean) / c_std
        stylized = normalized * s_std + s_mean
        return alpha * stylized + (1 - alpha) * content_feat

    def _calc_content_loss(self, input, target):
        #Fix shape unmatch issue
        if input.shape != target.shape:
            # print(f"Input and target shape not match for content loss {input.shape, target.shape}")
            input = nn.functional.interpolate(input, size=(target.shape[2],target.shape[3]), mode="bilinear", align_corners=False)
        return self.mse_loss(input, target)

    def _calc_style_loss(self, input, target):
        #Fix shape unmatch issue
        if input.shape != target.shape:
            # print(f"Input and target shape not match for style loss {input.shape, target.shape}")
            input = nn.functional.interpolate(input, size=(target.shape[2],target.shape[3]), mode="bilinear", align_corners=False)
        input_mean, input_std = self._calc_mean_std(input)
        target_mean, target_std = self._calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)


    def train(self, train_content_loader, train_style_loader, path_name,learn_rate = 1e-4, steps=160000, lr_decay = 5e-5, content_weight = 1.0, style_weight=1.0, log_interval = 500, epoch = 1):
        #freeze encoder
        self.encoder.to(self.device).eval()
        
        #Train decoder
        self.decoder.to(self.device).train()

        #enable content and style image to be cycled through
        contents = cycle(train_content_loader)
        styles = cycle(train_style_loader)

        total_content_loss = 0
        total_style_loss = 0
        total_total_loss = 0
        
        optimizer = torch.optim.Adam(self.decoder.parameters(), lr=learn_rate)
        for step in range(1,steps+1):
            #Loop the images to prevent images from running out
            content, content_label = next(contents)
            style, style_label = next(styles)

            #Transfer to available gpu
            content = content.to(self.device)
            style = style.to(self.device)
            
            lr = learn_rate / (1.0 + lr_decay * (step + ((epoch -1) * steps)))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

           
            #Encode image
            n_c, (n_s,) = self.encodeAll(content, [style])
            #Transformer
            features = self._adain(n_c,n_s)
            
            #Decode
            output = self.decoder(features)
    
            #Enocde the output again
            newOutput = self.encoder(output)
    
            #Calculate content and style loss
            content_loss = self._calc_content_loss(newOutput, features) * content_weight
            style_loss = self._calc_style_loss(newOutput, n_s) * style_weight
    
            #Total loss
            loss = content_loss+style_loss

            total_content_loss += content_loss
            total_style_loss += style_loss
            total_total_loss += loss
            
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            if step % log_interval == 0 and step != steps:
                #Average losses
                print(f"[{step}/{steps}] content={total_content_loss/log_interval:.4f} style={total_style_loss/log_interval:.4f} total={total_total_loss/log_interval:.6f}")
                #Clear 
                total_content_loss = 0
                total_style_loss = 0
                total_total_loss = 0
                #Clear cache to prevent crash
                if self.device.type == "mps":
                    torch.mps.empty_cache()

        torch.save(self.decoder.state_dict(), path_name)
        return total_content_loss/log_interval, total_style_loss/log_interval, total_total_loss/log_interval

    def epochTraining(self, train_content_loader, train_style_loader, path_name, epoch=5, log_interval=1000, steps = 50000,content_weight = 1.0, style_weight=1.0, earlystopCallback = None, learn_rate = 1e-4, lr_decay = 5e-5,):
        for e in range(1, epoch+1):
            print(f"Epoch: {e}")
            if self.prebuild_decoder is not None:
                self.loadPrebuildDecoder()

            #extract final output
            cl, sl,tl = self.train(train_content_loader, train_style_loader, path_name= f"{path_name}_{e}.pth", steps=steps, content_weight = content_weight, style_weight=style_weight, log_interval=log_interval, epoch = e, learn_rate = learn_rate, lr_decay=lr_decay)

            #print last run
            print(f"[{steps}/{steps}] content={cl:.2f} style={sl:.2f} total={tl:.3f}")
            
            #Load new decoder
            self.prebuild_decoder = f"{path_name}_{e}.pth"

            #terminate if true
            if earlystopCallback is not None:
                if earlystopCallback(cl,sl,tl):
                    break
                
        
    def stylisation(self, contentFeature, styleFeatures, alpha = 1.0):
        adaINOut = self._adain(contentFeature, styleFeatures[0], alpha) #Handle 1 style currently
        return adaINOut

        
    def spatialControl(self, foreground_style_index = None, background_style_index = None):

        #Remove background and alpha
        self.content.removeBackground()
        self.content.extractAlpha()
        content = self.tensorisedContent
        styles = self.tensorisedStyles
        dimension = content.dim()

        #tensorise mask
        foreMask = torch.tensor(self.content.alphaOutput(), dtype=torch.float32, device = self.device)

        #Add dimension if insufficient
        while foreMask.dim() < content.dim():
            foreMask = foreMask.unsqueeze(0)
        
        #Mask for style and content
        foreStyle = (styles[foreground_style_index]) if foreground_style_index != None else None
        backStyle = (styles[background_style_index]) if background_style_index != None else None

        #Extract Features
        contentFeature = self.encoder(content)
        foreFeatureStyle = self.encoder(foreStyle) if foreground_style_index != None else None
        backFeatureStyle = self.encoder(backStyle) if background_style_index != None else None

        #Apply Adain
        styledForeground = self.stylisation(contentFeature, [foreFeatureStyle]) if foreFeatureStyle != None else contentFeature
        styledBackground = self.stylisation(contentFeature, [backFeatureStyle]) if backFeatureStyle != None else contentFeature

        #Resize mask
        d1,d2,h,w = contentFeature.shape
        foreMaskFeature = nn.functional.interpolate(foreMask, size=(h,w), mode="bilinear", align_corners=False)
        backMaskFeature = 1 - foreMaskFeature

        #Combine - prevent leak using masks
        combinedFeature = foreMaskFeature * styledForeground + backMaskFeature * styledBackground

        self.decoding(combinedFeature)

    #Call pipeline and test time
    def pipeline(self, alpha = 1.0):
        self.start = time.perf_counter()
        #Encode image
        n_c,n_s = self.encodeAll(self.tensorisedContent,self.tensorisedStyles)
        #Transformer
        features = self.stylisation(n_c,n_s, alpha)
        #Decode
        self.decoding(features)
        self.end = time.perf_counter()