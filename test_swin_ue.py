import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch,cv2,warnings,argparse
from tqdm import tqdm
from torch import nn
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
from SWIN_UE import swin_base_patch4_window7_224

from datasets.dataset import MyDataset
warnings.filterwarnings('ignore')
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# torch.cuda.set_device(3)
torch.set_num_threads(2)
num_classes = 2
batch_size = 64
EPOCH = 100
pre_epoch = 0  
input_size = 224

class MainModel(nn.Module):
    def __init__(self,pretrained=True,pertubrations=(False,False,False,False),uncertainty=0.5):
        super(MainModel,self).__init__()
        self.pertubrations = pertubrations
        self.uncertainty = uncertainty
        self.model = swin_base_patch4_window7_224(pretrained=pretrained,pertubrations=pertubrations,uncertainty=uncertainty,num_classes=2)
        # for name, param in self.model.named_parameters():
        #     print(name,'-->',param.type(),'-->',param.dtype,'-->',param.shape)

    def forward(self, x,mask=None,labels=None):
        x = self.model.forward(x,mask,labels)
        return x

parser = argparse.ArgumentParser(description='PyTorch DeepNetwork Training')
parser.add_argument('--outf', default='./model/swin/base_ue_s4', help='folder to output images and model checkpoints')  # ��Ӝ�X�
args = parser.parse_args()
if not os.path.exists(args.outf):
    os.makedirs(args.outf)

def test_all():
    LR = 3e-5  
    print("Start Testing, DeepNetwork!")  

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

    net = MainModel(pretrained=False,pertubrations=(False,False,False,True),uncertainty=0.5)
    net = net.cuda()


    with open(os.path.join(args.outf,"new_acc.txt"), "w") as f:
        for epoch in range(pre_epoch, 42):
            path = path = r'/home/Laizhm/proj/swin_ue/model/swin/base_ue_s4/net_%03d.pth' % (epoch+1)
            net.load_state_dict(torch.load(path,map_location='cuda:0'))
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
                        images, label = images.cuda(),  label.cuda()
                        outputs = net(images)

                        _, predicted = torch.max(outputs.data, 1)
                        total += label.size(0)
                        correct += (predicted == label).cpu().sum()
                        pre.extend(((F.softmax(outputs, dim=1)[:, 1]).cpu()).numpy())

                    acc = 100. * float(correct) / float(total)
                    auc = 100. * roc_auc_score(labels, pre)
                    print('%s|acc:%.3f%%, auc:%.3f%%' % (val_name,acc,auc))
                    f.write("%s|acc:%.3f%%, auc:%.3f%%; " % (val_name,acc,auc))

                f.write('\n')
                f.flush()
        print("Testing Finished, TotalEPOCH=%d" % EPOCH)
def test():
    # print(torch.cuda.current_device())
    # print(torch.cuda.device_count())
    print("Start Testing, DeepNetwork!")  
    val_set_d = MyDataset(txt_path=r'/home/Laizhm/proj/swin_ue/datasets/txt/dfdc_test_lip.txt')
    val_loader_d = torch.utils.data.DataLoader(val_set_d, batch_size=64, shuffle=True)

    net = MainModel(pretrained=False,pertubrations=(False,False,False,True),uncertainty=0.5)

    net = net.cuda()

    with open(os.path.join(args.outf,"dfdc_acc.txt"), "w+") as f:
        for epoch in range(0, 42):
            path = path = r'/home/Laizhm/proj/swin_ue/model/swin/base_ue_s4/net_%03d.pth' % (epoch+1)
            net.load_state_dict(torch.load(path,map_location='cuda:0'))
            with torch.no_grad():
                net.eval()
                f.write("EPOCH=%03d: " % (epoch + 1))
                correct = 0
                total = 0
                labels = []
                pre = []
                for data in tqdm(val_loader_d):
                    images,label = data
                    labels.extend(label)
                    images, label = images.cuda(),  label.cuda()
                    outputs = net(images)

                    _, predicted = torch.max(outputs.data, 1)
                    total += label.size(0)
                    correct += (predicted == label).cpu().sum()
                    pre.extend(((F.softmax(outputs, dim=1)[:, 1]).cpu()).numpy())

                acc = 100. * float(correct) / float(total)
                auc = 100. * roc_auc_score(labels, pre)
                print('%s|acc:%.3f%%, auc:%.3f%%' % ('dfdc',acc,auc))
                f.write("%s|acc:%.3f%%, auc:%.3f%%; " % ('dfdc',acc,auc))

                f.write('\n')
                f.flush()
        print("Testing Finished, TotalEPOCH=%d" % EPOCH)
if __name__ == "__main__":
    test()