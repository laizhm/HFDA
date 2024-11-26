import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch,cv2,warnings,argparse
from tqdm import tqdm
from torch import nn
import torch.optim as optim
import argparse
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
from SWIN import swin_base_patch4_window7_224
import random
import numpy as np

from MyDataset import MyDataset,MyDataset_patch_mask_F,MyDataset_patch_mask_R,albu_transforms
warnings.filterwarnings('ignore')
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
torch.set_num_threads(2)
num_classes = 2
batch_size = 64
EPOCH = 50
pre_epoch = 0
input_size = 224

class MainModel(nn.Module):
    def __init__(self,pretrained=True):
        super(MainModel,self).__init__()
        self.model = swin_base_patch4_window7_224(pretrained=pretrained,num_classes=2)
        # for name, param in self.model.named_parameters():
        #     print(name,'-->',param.type(),'-->',param.dtype,'-->',param.shape)

    def forward(self, x):
        x = self.model.forward(x)
        return x

parser = argparse.ArgumentParser(description='PyTorch DeepNetwork Training')
parser.add_argument('--outf', default='./swin/base_no_aug', help='folder to output images and model checkpoints')  # 
args = parser.parse_args()
if not os.path.exists(args.outf):
    os.makedirs(args.outf)

def main():
    LR = 3e-5  
    print("Start Training, DeepNetwork!")  
    training_set_f = MyDataset_patch_mask_F(txt_path=r'/home/Laizhm/proj/swin_ue/datasets/txt/ffpp_train_split_8.txt',transforms=None)#Face2Face,Deepfakes,FaceSwap,NeuralTextures
    training_generator_f = torch.utils.data.DataLoader(training_set_f,batch_size=64,shuffle=True)

    training_set_r = MyDataset_patch_mask_R(txt_path=r'/home/Laizhm/proj/swin_ue/datasets/txt/ffpp_train_split_8.txt',transforms=None)
    training_generator_r = torch.utils.data.DataLoader(training_set_r,batch_size=16,shuffle=True)

    val_set_ffpp = MyDataset(txt_path=r'/home/Laizhm/proj/swin_ue/datasets/txt/ffpp_test_split.txt')
    val_loader_ffpp = torch.utils.data.DataLoader(val_set_ffpp, batch_size=batch_size, shuffle=True)
    
    val_set_cd2 = MyDataset(txt_path=r'/home/Laizhm/proj/swin_ue/datasets/txt/CD2_test.txt')
    val_loader_cd2 = torch.utils.data.DataLoader(val_set_cd2, batch_size=batch_size, shuffle=True)
    val_set_d = MyDataset(txt_path=r'/home/Laizhm/proj/swin_ue/datasets/txt/dfdc_test_lip.txt')
    val_loader_d = torch.utils.data.DataLoader(val_set_d, batch_size=batch_size, shuffle=True)
    val_set_cd1 = MyDataset(txt_path=r'/home/Laizhm/proj/swin_ue/datasets/txt/CD1_test.txt')
    val_loader_cd1 = torch.utils.data.DataLoader(val_set_cd1, batch_size=batch_size, shuffle=True)
    val_set_dp = MyDataset(txt_path=r'/home/Laizhm/proj/swin_ue/datasets/txt/dfdcp_test.txt')
    val_loader_dp = torch.utils.data.DataLoader(val_set_dp, batch_size=batch_size, shuffle=True)
    val_set_dfr = MyDataset(txt_path=r'/home/Laizhm/proj/swin_ue/datasets/txt/DFR_test.txt')
    val_loader_dfr = torch.utils.data.DataLoader(val_set_dfr, batch_size=batch_size, shuffle=True)

    val_loaders = [val_loader_ffpp,val_loader_cd1,val_loader_cd2,val_loader_d,val_loader_dp,val_loader_dfr]
    val_names = ['ffpp','cd1','cd2','dfdc','dfdcp','dfr']

    net = MainModel(pretrained=True)
    device = torch.device("cuda:0")

    net = net.to(device)

    # criterion
    criterion = nn.CrossEntropyLoss()
    # optimizer
    optimizer = optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-9)
    # optimizer = optim.SGD(net.parameters(), lr=LR,momentum=0.9, weight_decay=5e-4)
    # scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.35, min_lr=1e-6,patience=2, verbose=True)
    with open(os.path.join(args.outf,"acc.txt"), "w") as f:
        with open(os.path.join(args.outf,"log.txt"), "w")as f2:
            for epoch in range(pre_epoch, EPOCH):
                # scheduler.step(epoch)
                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                length = len(training_generator_f)
                for i, data in enumerate(zip(training_generator_f,training_generator_r), 0):

                    data_f,data_r = data
                    input_f, target_f = data_f
                    input_f, target_f = input_f.to(device), target_f.to(device)
                    input_r, target_r = data_r
                    input_r, target_r = input_r.to(device), target_r.to(device)
                    input = torch.cat([input_f,input_r],dim=0)
                    target = torch.cat([target_f,target_r],dim=0)
                    # ��
                    optimizer.zero_grad()
                    # forward + backward

                    output = net(input)

                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

                    sum_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += predicted.eq(target.data).cpu().sum()
                    print('[epoch:%d, iter:%d/%d] Lr: %.6f | Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), (epoch+1) * length, optimizer.param_groups[0]['lr'], sum_loss / (i + 1),
                             100. * float(correct) / float(total)))
                    f2.write('%03d  %05d/%d  Lr: %.6f |Loss: %.03f | Acc: %.3f%% '
                             % (epoch + 1, (i + 1 + epoch * length), (epoch+1) * length, optimizer.param_groups[0]['lr'], sum_loss / (i + 1),
                                100. * float(correct) / float(total)))
                    f2.write('\n')
                    f2.flush()
                    # break
                scheduler.step(sum_loss/length)

                print("Waiting Test!")
                with torch.no_grad():
                    net.eval()
                    f.write("EPOCH=%03d" % (epoch + 1))
                    for val_name,val_loader in zip(val_names,val_loaders):
                        correct = 0
                        total = 0
                        labels = []
                        pre = []
                        for data in tqdm(val_loader):
                            images,label = data
                            labels.extend(label)
                            images, label = images.to(device),  label.to(device)
                            outputs = net(images)

                            _, predicted = torch.max(outputs.data, 1)
                            total += label.size(0)
                            correct += (predicted == label).cpu().sum()
                            pre.extend(((F.softmax(outputs, dim=1)[:, 1]).cpu()).numpy())
                            # break
                        # print(pre)
                        acc = 100. * float(correct) / float(total)
                        auc = 100. * roc_auc_score(labels, pre)
                        print('%s|acc:%.3f%%, auc:%.3f%%' % (val_name,acc,auc))
                        f.write("%s|acc:%.3f%%, auc:%.3f%%; " % (val_name,acc,auc))

                    f.write('\n')
                    f.flush()
                    print('Saving model......')
                    torch.save(net.state_dict(), '%s/net_%03d.pth' % (args.outf, epoch + 1))
            print("Training Finished, TotalEPOCH=%d" % EPOCH)

if __name__ == "__main__":
    seed = 111
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main()