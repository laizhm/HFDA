from torch.utils.data import Dataset
import cv2,torch,torchvision
import numpy as np
from albumentations import Compose, RandomBrightnessContrast, \
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, \
    ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur,Resize,ISONoise
from albumentations.pytorch.functional import img_to_tensor
from PIL import Image
num_classes = 2
batch_size = 48

input_size = 224

class MyDataset(Dataset):
    def __init__(self, txt_path, size=(224,224),transforms = None,expected_method=None,exception_method = None, pasta=None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            if exception_method==None:
                if expected_method==None:
                    imgs.append((words[0], int(words[1])))
                else:
                    if 'original_sequences' in words[0] or expected_method in words[0]:
                        imgs.append((words[0], int(words[1])))
            else:
                if 'original_sequences' in words[0] or exception_method not in words[0]:
                    imgs.append((words[0], int(words[1])))

        print(len(imgs))
        # random.shuffle(imgs)
        self.imgs = imgs
        self.size = size
        self.transforms = transforms
        self.normalize = {"mean": [0.5, 0.5, 0.5],"std": [0.5, 0.5, 0.5]},
        self.pasta = pasta
        # self.normalize = {"mean": [0.4989538, 0.3954232, 0.3735058],"std": [0.2393493, 0.1973593, 0.19328721]},

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        # image = cv2.imread(fn, cv2.IMREAD_COLOR)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.resize(image, self.size)
        image = Image.open(fn).convert("RGB")
        image = image.resize((224,224))
        if self.pasta:
            image = self.pasta(image)
        image = np.array(image)

        if self.transforms :
            data1 = self.transforms(image=image)
            image = data1['image']
        image = img_to_tensor(image, self.normalize[0])

        return image,  label
    def __len__(self):
        return len(self.imgs)
class MyDataset_patch_mask_F(Dataset):
    def __init__(self, txt_path, transforms = None,size=(224,224),expected_method=None,exception_method=None,mask_flag=False,data_len = 5744, pasta=None):#data_len:5744,21540
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            # if method in words[0]:
            if exception_method==None:
                if 'original_sequences' not in words[0]:
                    if expected_method==None:
                        imgs.append((words[0], int(words[1]),words[0].replace('c23','masks')))
                    else:
                        if expected_method in words[0]:
                            imgs.append((words[0], int(words[1]),words[0].replace('c23','masks')))
            else:
                if exception_method not in words[0] and 'original_sequences' not in words[0]:
                    imgs.append((words[0], int(words[1]),words[0].replace('c23','masks')))
        
        # random.shuffle(imgs)
        if expected_method==None and exception_method==None:
            self.imgs = imgs[:data_len*4]
        else: 
            if expected_method!=None:
                self.imgs = imgs[:data_len]
            else: 
                if exception_method!=None:
                    self.imgs = imgs[:data_len*3]
        # self.imgs = imgs
        self.transforms = transforms
        self.mask_flag = mask_flag
        self.size = size
        self.normalize = {"mean": [0.5, 0.5, 0.5],"std": [0.5, 0.5, 0.5]},
        self.pasta = pasta
        # self.normalize = {"mean": [0.4989538, 0.3954232, 0.3735058],"std": [0.2393493, 0.1973593, 0.19328721]},
        print(len(self.imgs))

    def __getitem__(self, index):
        fn, label,maskn = self.imgs[index]
        image = Image.open(fn).convert("RGB")
        image = image.resize((224,224))
        
        aug_image = self.pasta(image)
        image = np.array(image)
        aug_image = np.array(aug_image)

        mask = cv2.imread(maskn,0)
        mask = cv2.resize(mask, self.size)
        patchlabel = generate_patchlabel(mask,label,7,7)
        patchlabel = patchlabel.flatten()

        if self.transforms :
            data1 = self.transforms(image=image)
            data2 = self.transforms(image=aug_image)
            image = data1['image']
            aug_image = data2['image']
        image = img_to_tensor(image, self.normalize[0])
        aug_image = img_to_tensor(aug_image, self.normalize[0])
        return image, aug_image, label, label, patchlabel, patchlabel
    
    def __len__(self):
        return len(self.imgs)
class MyDataset_patch_mask_R(Dataset):
    def __init__(self, txt_path, transforms = None,size=(224,224),mask_flag=False, pasta=None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            if 'original_sequences' in words[0]:
                imgs.append((words[0], int(words[1]),words[0].replace('faces50','faces50_mask')))
        # random.shuffle(imgs)
        self.imgs = imgs
        self.mask_flag = mask_flag
        self.transforms = transforms
        self.size = size
        self.normalize = {"mean": [0.5, 0.5, 0.5],"std": [0.5, 0.5, 0.5]},
        self.pasta = pasta
        # self.normalize = {"mean": [0.4989538, 0.3954232, 0.3735058],"std": [0.2393493, 0.1973593, 0.19328721]},
        print(len(self.imgs))

    def __getitem__(self, index):
        fn, label,maskn = self.imgs[index]
        image = Image.open(fn).convert("RGB")
        image = image.resize((224,224))
        
        aug_image = self.pasta(image)
        image = np.array(image)
        aug_image = np.array(aug_image)

        mask = cv2.imread(maskn,0)
        mask = cv2.resize(mask, self.size)
        patchlabel = generate_patchlabel(mask,label,7,7)
        patchlabel = patchlabel.flatten()

        if self.transforms :
            data1 = self.transforms(image=image)
            data2 = self.transforms(image=aug_image)
            image = data1['image']
            aug_image = data2['image']
        image = img_to_tensor(image, self.normalize[0])
        aug_image = img_to_tensor(aug_image, self.normalize[0])
        return image, aug_image, label, label, patchlabel, patchlabel
    
    def __len__(self):
        return len(self.imgs)
    
def albu_transforms(size=224):
    return Compose([
                ImageCompression(quality_lower=70, quality_upper=100, p=0.3),
                GaussNoise(p=0.5),
                GaussianBlur(blur_limit=3, p=0.5),
                ISONoise(p=0.5),
                HorizontalFlip(0.5),
                OneOf([
                    HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=0.3),
                    FancyPCA(),
                    RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1,0.1)),
                ])], p=0.8)
def create_val_transforms(size=224):
    return Compose([
        Resize(size, size, p=1),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
    ])

def generate_patchlabel(input_, label,row, col):
    """print patch label or score list

    Args:
        input_ (list): patch label or score list
        row (int): row of patch
        col (int): col of patch
    """
    x,y = input_.shape
    p_label = np.zeros([row,col])
    x,y = x//row,y//col
    for r in range(row):
        for c in range(col):
            if np.max(input_[r * x:r * x+x,c*y:c*y+y])>=1.0:
                p_label[r,c]=1.0
                # if label==1:
                #     p_label[r,c]=1.0
                # else:
                #     p_label[r,c]=2.0
    return p_label

def main():
    training_set_f = MyDataset_patch_mask_F(txt_path=r'/home/Laizhm/proj/swin_ue/datasets/txt/2023/ffpp_train_split_c0.txt',transforms=albu_transforms(),expected_method='NeuralTextures')#Face2Face,Deepfakes,FaceSwap,NeuralTextures
    training_generator_f = torch.utils.data.DataLoader(training_set_f,batch_size=16,shuffle=True)

    training_set_r = MyDataset_patch_mask_R(txt_path=r'/home/Laizhm/proj/swin_ue/datasets/txt/2023/ffpp_train_split_c0.txt',transforms=albu_transforms())
    training_generator_r = torch.utils.data.DataLoader(training_set_r,batch_size=16,shuffle=True)

    val_set_f2f = MyDataset(txt_path=r'/home/Laizhm/proj/swin_ue/datasets/txt/2023/ffpp_test_split_c0.txt',expected_method='Face2Face')
    val_loader_f2f = torch.utils.data.DataLoader(val_set_f2f, batch_size=batch_size, shuffle=True)
    val_set_df = MyDataset(txt_path=r'/home/Laizhm/proj/swin_ue/datasets/txt/2023/ffpp_test_split_c0.txt',expected_method='Deepfakes')
    val_loader_df = torch.utils.data.DataLoader(val_set_df, batch_size=batch_size, shuffle=True)
    val_set_fs = MyDataset(txt_path=r'/home/Laizhm/proj/swin_ue/datasets/txt/2023/ffpp_test_split_c0.txt',expected_method='FaceSwap')
    val_loader_fs = torch.utils.data.DataLoader(val_set_fs, batch_size=batch_size, shuffle=True)
    val_set_nt = MyDataset(txt_path=r'/home/Laizhm/proj/swin_ue/datasets/txt/2023/ffpp_test_split_c0.txt',expected_method='NeuralTextures')
    val_loader_nt = torch.utils.data.DataLoader(val_set_nt, batch_size=batch_size, shuffle=True)
    val_set_ffpp = MyDataset(txt_path=r'/home/Laizhm/proj/swin_ue/datasets/txt/2023/ffpp_test_split_c0.txt')
    val_loader_ffpp = torch.utils.data.DataLoader(val_set_ffpp, batch_size=batch_size, shuffle=True)

    val_set_c = MyDataset(txt_path=r'/home/Laizhm/proj/swin_ue/datasets/txt/2023/CD2_test.txt')
    val_loader_c = torch.utils.data.DataLoader(val_set_c, batch_size=batch_size, shuffle=True)
    val_set_d = MyDataset(txt_path=r'/home/Laizhm/proj/swin_ue/datasets/txt/2023/dfdc_test_lip.txt')
    val_loader_d = torch.utils.data.DataLoader(val_set_d, batch_size=batch_size, shuffle=True)
    val_set_cd = MyDataset(txt_path=r'/home/Laizhm/proj/swin_ue/datasets/txt/2023/CD1_test.txt')
    val_loader_cd = torch.utils.data.DataLoader(val_set_cd, batch_size=batch_size, shuffle=True)
    val_set_dp = MyDataset(txt_path=r'/home/Laizhm/proj/swin_ue/datasets/txt/2023/dfdcp_test.txt')
    val_loader_dp = torch.utils.data.DataLoader(val_set_dp, batch_size=batch_size, shuffle=True)
    val_set_dfr = MyDataset(txt_path=r'/home/Laizhm/proj/swin_ue/datasets/txt/2023/DFR_test.txt')
    val_loader_dfr = torch.utils.data.DataLoader(val_set_dfr, batch_size=batch_size, shuffle=True)

    val_loaders = [val_loader_f2f,val_loader_df,val_loader_fs,val_loader_nt,val_loader_cd,val_loader_c,val_loader_d,val_loader_dp,val_loader_dfr]
    val_names = ['f2f','df','fs','nt','cd1','cd2','dfdc','dfdcp','dfr']

if __name__ == "__main__":
    main()