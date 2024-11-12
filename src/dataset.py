from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile

import os
from glob import glob
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as F
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

class SquarePad:

    def __init__(self, patch_size) -> None:
        self.max_patch = max(patch_size[0], patch_size[1])

    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        max_wh = max(max_wh, self.max_patch)
        hp = int((max_wh - w) / 2)+1
        vp = int((max_wh - h) / 2)+1
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, 'constant')


class VimeoDatasets(Dataset):

    def __init__(self, data_dir, text_dir , image_size=256, NUM_IMAGES = 3000):
        self.text_dir  = text_dir # tri_trainlist.txt
        self.data_dir = data_dir #vimeo_arod/
        self.image_size = image_size
        self.image_path = [] #sorted(glob(os.path.join(self.data_dir, "*.*")))
        self.total_dir = os.path.join(self.data_dir,"sequences") #data_dir/sequences
        
        file = open(os.path.join(self.data_dir,self.text_dir),"r")
        lines = file.readlines()
        
        for index, line in enumerate(lines):

            if index > NUM_IMAGES + 1:
                break
            c = line.strip()
            tmp = os.path.join(self.data_dir,c)
            d = [os.path.join(tmp,f) for f in os.listdir(tmp)]
            self.image_path += d
        
        file.close()
        self.image_path = self.image_path[:NUM_IMAGES]
        print("lunghezza del dataset: ",len(self.image_path))
 
        #self.image_path = sorted(glob(os.path.join(self.data_dir, "*.*")))
        
    def __getitem__(self, item):
        image_ori = self.image_path[item]

        #image = cv2.imread(image_ori)

        image = Image.open(image_ori).convert('RGB')

        transform = transforms.Compose(
            [transforms.RandomCrop(self.image_size), transforms.ToTensor()]
        )
        return transform(image)
        
    def __len__(self):
        return len(self.image_path)


class TestKodakDataset(Dataset):
    
    def __init__(self, data_dir, image_size = 256, crop = True, random_crop = False, get_img_name=False):
        self.data_dir = data_dir
        self.image_size = image_size 
        self.crop = crop
        self.random_crop = random_crop
        self.get_img_name = get_img_name
        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")
        self.image_path = sorted(glob(os.path.join(self.data_dir, "*.*")))
        print(f'Random crop: {self.random_crop}')

    def __getitem__(self, item):
        image_ori = self.image_path[item]
        image = Image.open(image_ori).convert('RGB')

        if(self.crop):
            transform = transforms.Compose(
            [transforms.CenterCrop(self.image_size), transforms.ToTensor()])
            
            if(self.random_crop):
                
                transform = transforms.Compose(
                [transforms.RandomCrop(self.image_size), transforms.ToTensor()])

        else:
            
            transform = transforms.Compose([transforms.ToTensor()])

        if(self.get_img_name):
            return transform(image), image_ori
        
        return transform(image)

    def __len__(self):
        return len(self.image_path)