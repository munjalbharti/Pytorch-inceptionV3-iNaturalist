from torch.utils.data import Dataset
import cv2
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
            x = self.flip_aug(self.pil_image (x))
            x = self.resize(x)
        else:
            x = self.resize(self.pil_image(x))

        x = self.tensor_aug(x) #converts image to 0 and 1 and to tensor, also chnages to C*h*w
        x = (x-0.5)*2 #since values are between 0 and 1, this operation makes it between [-1,1] as suggested by orignal paper

        return x,y


