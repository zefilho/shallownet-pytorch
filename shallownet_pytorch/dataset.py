import os
import cv2
import numpy as np

from dataclasses import dataclass
import torchvision.transforms as transforms

@dataclass 
class SimplePreprocessor:
    width: int
    height: int
    inter = cv2.INTER_AREA
 
    def preprocess(self, image):
        #resize image
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)

@dataclass
class ImageToTensor:
    dataformat: str =  None

    def preprocess(self, image):
        # apply the utility function that correctly rearranges the dimensions of the image
        # the image to torch tensor
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        # Convert the image to Torch tensor
        return transform(image)


class SimpleDataLoader:
    def __init__(self, preprocessors=None):
        # store the image preprocessor
        self.preprocessors = preprocessors

        # empty list
        self.preprocessors = preprocessors if preprocessors is not None else []
    
    def load(self, imagePaths, verbose=-1):
        # initialize list
        data, labels = [], []

        for (i, imgPath) in enumerate(imagePaths):
            # /path/to/dataset/{class}/{image}.jpg
            image = cv2.imread(imgPath)
            if image is None:
                print(f"[WARNING] imagem não pôde ser lida: {imgPath}")
                continue
            
            label = imgPath.split(os.path.sep)[-2]
            # check to see if our preprocessors are not None
            if self.preprocessors is not None:
                # loop over the preprocessors and apply each to the image
                for p in self.preprocessors:
                    image = p.preprocess(image)
            # treat our processed image as a "feature vector"
            # by updating the data list followed by the labels
            data.append(image)
            labels.append(label)
        
            # show an update every ‘verbose‘ images
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1, len(imagePaths)))

        # return a tuple of the data and labels
        return (np.array(data), np.array(labels))
    

if __name__ == "__main__":
    sp = SimplePreprocessor(32, 32)
    iap = ImageToTensor()
    print('ok')
