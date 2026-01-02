import torch
import lpips

class LPIPS:
    def __init__(self, mode="alex"):
        self.model = lpips.LPIPS(net = mode)

    def _normalise(self, img):
        #convert range to [-1,1] from [0,1]
        return img * 2 - 1

    def eval(self, inputImg, outputImg):

        #Normalise
        inputImg = self._normalise(inputImg)
        outputImg = self._normalise(outputImg)

        #Scoring
        score  = self.model.forward(inputImg, outputImg)

        #Average the score
        return score.mean().detach().item()