import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import cv2
import numpy as np
import torch

from pytorch_grad_cam import GradCAM,ScoreCAM,GradCAMPlusPlus,AblationCAM,XGradCAM,EigenCAM,EigenGradCAM,LayerCAM,FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image,preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.ablation_layer import AblationLayerVit

from train_swin_fda_nief_ablation_v3 import MainModel

def reshape_transform(tensor,height=7,width=7):
    print(tensor.shape)
    result = tensor.reshape(tensor.size(0),height,width,tensor.size(2))
    result = result.transpose(2,3).transpose(1,2)
    return result

def generate_cam(image_path,tail='dsm',mtype=None):
    model = MainModel(pretrained=True)
    path = '/home/Laizhm/proj/FDA/ablation/fda_222_nief_alpha_0.5_r4/net_011.pth'
    model.load_state_dict(torch.load(path))
    model.eval()
    model = model.cuda()
    target_layers=[model.model.norm]
    # 创建 GradCAM 对象
    cam = GradCAM(model=model,target_layers=target_layers,reshape_transform=reshape_transform)
    # 读取输入图像

    bgr_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    bgr_img = cv2.resize(bgr_img, (224, 224))
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    rgb_img = np.float32(rgb_img)/255
    dir,img_name = os.path.split(image_path)
    name = os.path.split(dir)[-1]+'_'+img_name[:-4]
    if mtype!=None:
        name = mtype + '_' + name
    # print(name)


    input_tensor = preprocess_image(rgb_img,mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
    # input_tensor = input_tensor.float()


    # input_tensor = input_tensor.cuda()
    #print(input_tensor.shape)
    #grad-cam
    nt = 0
    target_category = [ClassifierOutputTarget(nt)] #
    #target_category = None
    grayscale_cam = cam(input_tensor=input_tensor,targets=target_category)
    grayscale_cam = grayscale_cam[0,:]


    visualization = show_cam_on_image(rgb_img, grayscale_cam)


    # cv2.imwrite('cam.jpg', visualization)
    # cv2.imwrite(f"/home/zhangj/proj/vpt/dsm_img/cam_real/dfdcp/{name}.png",bgr_img)
    # cv2.imwrite(f"/home/zhangj/proj/vpt/dsm_img/cam_real/dfdcp/{name}_{tail}_cam.png",visualization)
    # cv2.imwrite(f"/home/zhangj/proj/vpt/dsm_img/cam_real/cd2/{name}.png",bgr_img)
    # cv2.imwrite(f"/home/zhangj/proj/vpt/dsm_img/cam_real/cd2/{name}_{tail}_cam_{nt}.png",visualization)
    cv2.imwrite(f"/home/zhangj/proj/FDA/{name}.png",bgr_img)
    cv2.imwrite(f"/home/zhangj/proj/FDA/{name}_{tail}_cam.png",visualization)

if __name__ == "__main__":

    # image_path = "/PublicFile/zhj/DFDC-P/test_set_23/original_videos/1892339_A_001/45.png"
    # image_path = f"/PublicFile/zhj/Celeb-DF-v2/faces23/id1_0002/{32}.png"
    # model = train_dsm_AGHS.MainModel(pretrained=False)
    # path = r'/home/zhangj/proj/vpt/results/dsm_AGHS_nograd_adamw_4e6/net_%03d.pth' % (10+1)
    # model.load_state_dict(torch.load(path))
    # model.eval()
    # model = model.cuda()
    # generate_cam(image_path,'dsm')
    # # torch.cuda.empty_cache()

    # model1 = train_vit.MainModel(pretrained=False)
    # # print(model1)
    # path1 = r'/home/zhangj/proj/vpt/results/vit_base_adamw_4e6/net_%03d.pth' % (18)
    # model1.load_state_dict(torch.load(path1))
    # model1.eval()
    # model1 = model1.cuda()
    # generate_cam(image_path,'base')
    # for i in range(9,10):
    #     for j in range(10):
    #         img_dir = f"/PublicFile/zhj/Celeb-DF-v2/faces23/id{i}_000{j}"
    #         if not os.path.exists(img_dir):
    #             continue
    #         imgs = os.listdir(img_dir)
    #         for img in imgs:
    #             image_path = os.path.join(img_dir,img)
    #             generate_cam(image_path,'dsm')
    #             generate_cam(image_path,'base')
    # root = '/PublicFile/zhj/DFDC-P/test_set/original_videos'
    ffroot = '/PublicFile/Laizhm/FaceForensic++_raw/manipulated_sequences/Deepfakes/c23/faces23'
    cd2root = '/PublicFile/Laizhm/Celeb-DF-v2/Celeb-synthesis/faces3_1'
    dir_names = os.listdir(ffroot)[-10:-5]
    for dir_name in dir_names:
        img_dir = os.path.join(ffroot,dir_name)
        imgs = os.listdir(img_dir)[:2]
        for img in imgs:
            image_path = os.path.join(img_dir,img)
            generate_cam(image_path,'fda',mtype='ff-fake')
            #generate_cam(image_path,'base',mtype='NT')

