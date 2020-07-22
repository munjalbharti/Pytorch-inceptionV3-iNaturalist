import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
import math
import random
from scipy.misc import imread
import imageio
import imgaug as ia
from imgaug import augmenters as iaa
from matplotlib import pyplot as plt
import numpy as np 
from torchvision import transforms

class CUB(Dataset):
    def __init__(self, files_path, labels, train_test, image_name, train=True, 
                 transform=False):
      
        self.files_path = files_path
        self.labels = labels
        self.transform = transform
        self.train_test = train_test
        self.image_name = image_name
        
        if train:
          mask = self.train_test.is_train.values == 1
          
        else:
          mask = self.train_test.is_train.values == 0
        
        
        self.filenames = self.image_name.iloc[mask]
        self.labels = self.labels[mask]
        self.num_files = self.labels.shape[0]

        # augmentation params
        self.im_size = [299, 299]  # can change this to train on higher res
        self.mu_data = [0.485, 0.456, 0.406] #imagent mean
        self.std_data = [0.229, 0.224, 0.225] #imagenet std
        self.brightness = 0
        self.contrast = 0
        self.saturation = 0
        self.hue = 0

        self.resize = transforms.Resize((self.im_size[0],self.im_size[1]))
        self.pil_image = transforms.ToPILImage()
        self.scale_aug = transforms.RandomResizedCrop(size=self.im_size[0]) #check documentation, this augmentation is normally used for inception network
        self.flip_aug = transforms.RandomHorizontalFlip()
        self.color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        self.tensor_aug = transforms.ToTensor()
        #self.norm_aug = transforms.Normalize(mean=self.mu_data, std=self.std_data)


    def read_image(self, path):
        im = cv2.imread(str(path))
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


    # Data Augmentation
    def center_crop(self,im, min_sz=None):
        """ Returns a center crop of an image"""
        r,c,*_ = im.shape
        if min_sz is None: min_sz = min(r,c)
        start_r = math.ceil((r-min_sz)/2)
        start_c = math.ceil((c-min_sz)/2)
        return self.crop(im, start_r, start_c, min_sz, min_sz)

    def crop(self,im, r, c, target_r, target_c): return im[r:r+target_r, c:c+target_c]

    def random_crop(self,x, target_r, target_c):
        """ Returns a random crop"""
        r,c,*_ = x.shape
        rand_r = random.uniform(0, 1)
        rand_c = random.uniform(0, 1)
        start_r = np.floor(rand_r*(r - target_r)).astype(int)
        start_c = np.floor(rand_c*(c - target_c)).astype(int)
        return self.crop(x, start_r, start_c, target_r, target_c)

    def rotate_cv(self,im, deg, mode=cv2.BORDER_REFLECT, interpolation=cv2.INTER_AREA):
        """ Rotates an image by deg degrees"""
        r,c,*_ = im.shape
        M = cv2.getRotationMatrix2D((c/2,r/2),deg,1)
        return cv2.warpAffine(im,M,(c,r), borderMode=mode, 
                              flags=cv2.WARP_FILL_OUTLIERS+interpolation)


    def apply_transforms(self, x, sz=299, train = True):
      """ Applies a random crop, rotation"""
      if train:

        while True:
          h_rand = int(np.random.uniform(1,x.shape[0]))
          w_rand = int(np.random.uniform(1,x.shape[1]))
          if h_rand == 0 or w_rand == 0: continue
          area = x.shape[0] * x.shape[1]
          if (w_rand/h_rand <= 1.33) and (w_rand/h_rand >= 0.75) and (w_rand*h_rand >= area*0.1)  and (w_rand*h_rand <= area):
            break

        seq = iaa.Sequential([
          iaa.size.CropToFixedSize(w_rand,h_rand),
          iaa.size.Resize(sz),
          iaa.Fliplr(0.5),
          iaa.color.AddToBrightness(),
          iaa.contrast.LinearContrast(),
          iaa.Grayscale(alpha=(0.0, 1.0)),
           ])
      else:
        seq = iaa.Sequential([
          iaa.size.Crop(percent=1-0.875),

          ])    

      im = seq(image=x)

      return im


    def normalize(self,im):
        """Normalizes images with Imagenet stats."""
        imagenet_stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
        return (im/255.0 - imagenet_stats[0])/imagenet_stats[1]       
    
    def __len__(self):
        return self.num_files

    def image_path_at(self, index):
        file_name = self.filenames.iloc[index, 1]
        path = self.files_path / 'images' / file_name
        return path


    def __getitem__(self, index):
        y = self.labels.iloc[index,1] - 1

        file_name = self.filenames.iloc[index, 1]
        path = self.files_path/'images'/file_name
        x = self.read_image(path)

        if self.transform:
            #x = self.scale_aug(self.pil_image (x)) #scale aug only works with pil image
            x = self.flip_aug(self.pil_image (x))
           # x = self.color_aug(x)

            x = self.resize(x)

            #x = self.apply_transforms(x) #only flip
            #x = x.astype(np.float32, copy=False)
            #x = x / 255
            #x = cv2.resize(x, (299, 299), interpolation=cv2.INTER_LINEAR)

            #x = cv2.resize(x,(299,299))
        else:
            x = self.resize(self.pil_image(x))
            #x = self.random_crop(x,299,299)
            # x = x.astype(np.float32, copy=False)
             #x = x / 255
            #x = self.center_crop(self.pil_image (x))

            #x = self.apply_transforms(x, train=False)


        x = self.tensor_aug(x) #converts image to 0 and 1 and to tensor, also chnages to C*h*w
        x = (x-0.5)*2 #since values are between 0 and 1, this operation makes it between [-1,1] as suggested by orignal paper





        return x,y


