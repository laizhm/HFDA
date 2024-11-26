import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch,cv2,warnings,argparse
from tqdm import tqdm
from torch import nn
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
from EER import compute_eer
from train_swin_fda_nief import MainModel

from MyDataset_withMask import MyDataset
warnings.filterwarnings('ignore')
torch.set_num_threads(2)
num_classes = 2
batch_size = 64
EPOCH = 100
pre_epoch = 0  
input_size = 224


parser = argparse.ArgumentParser(description='PyTorch DeepNetwork Training')
parser.add_argument('--outf', default='./fda_222_nief_00508', help='folder to output images and model checkpoints')  # dsm_AGHS_nograd_adamw_4e6
args = parser.parse_args()
if not os.path.exists(args.outf):
    os.makedirs(args.outf)

EPOCH = 50
def test_videos():
    txt_path = '/home/Laizhm/proj/swin_ue/Datasets/txt/2023/DFR_test.txt'#CD2_test, dfdcp_test, dfdc_test_lip,DFR_test
    out_txt_path = 'DFR_video_eer.txt'
    val_set = MyDataset(txt_path=txt_path)
    val_genenrator = torch.utils.data.DataLoader(val_set,batch_size=1,shuffle=False)
    # val_set_d = MyDataset(txt_path=r'/home/zhangj/code/dataset/dfdc_01.txt')
    # val_loader_d = torch.utils.data.DataLoader(val_set_d, batch_size=batch_size*8, shuffle=True)
    print(txt_path)
    fh = open(txt_path, 'r')
    dirs = []
    img_dirs = []
    lenght = 0
    label = None
    for line in fh:
        line = line.rstrip()
        words = line.split()
        if os.path.dirname(words[0]) not in dirs:
            if lenght!=0:
                img_dirs.append((dirs[-1],lenght,label))
            dirs.append(os.path.dirname(words[0]))
            lenght = 1
            label = int(words[1])
        else:
            lenght+=1
    img_dirs.append((dirs[-1],lenght,label))

    print(len(dirs),len(img_dirs))
    # print(dirs)
    

    #加载模型
    net = MainModel(pretrained=False)
    device = torch.device("cuda:0")
    f = open(os.path.join(args.outf, out_txt_path), "w")
    for i in range(100):
        path = args.outf+r'/net_%03d.pth' % (i+1)
        net.load_state_dict(torch.load(path))

        net = net.to(device)
        with torch.no_grad():
            correct = 0
            total = 0
            label = []
            pre = []
            dir1,lenght1,label1 = img_dirs[0]
            j=0
            predict = 0
            for data in tqdm(val_genenrator):
                net.eval()
                images,  labels = data
                images, labels = images.to(device),  labels.to(device)

                outputs = net(images)
                # 取得分最高的那个类 (outputs.data的索引号)
                # _, predicted = torch.max(outputs.data, 1)
                total +=1
                predict+=((F.softmax(outputs, dim=1)[:, 1]).cpu()).numpy()
                # print(predict)
                if total==lenght1:
                    pre.extend(predict/total)
                    label.extend([label1])
                    if (pre[-1]>0.5)==label1:
                        correct += 1
                    total = 0
                    predict = 0
                    if j+1<len(img_dirs):
                        dir1,lenght1,label1 = img_dirs[j+1]
                        j+=1
                #break 
                # print(roc_auc_score(label, pre))
            # correct += (pre>0.5).eq(label.data).cpu().sum()
            acc = 100. * float(correct) / float(len(label))
            auc = 100. * roc_auc_score(label, pre)
            eer = 100. * compute_eer(label, pre)

            print('EPOCH=%03d,auc：%.3f%%,eer：%.3f%%' % (i+1,auc,eer))
            f.write('EPOCH=%03d,auc：%.3f%%,eer：%.3f%%' % (i+1,auc,eer))
            f.write('\n')
            f.flush()
def test_all():
    print("Start Testing, DeepNetwork!")  

    val_set_ffpp = MyDataset(txt_path=r'/home/Laizhm/proj/swin_ue/datasets/txt/2023/ffpp_test_split.txt')
    val_loader_ffpp = torch.utils.data.DataLoader(val_set_ffpp, batch_size=batch_size, shuffle=True)

    val_set_cd2 = MyDataset(txt_path=r'/home/Laizhm/proj/swin_ue/datasets/txt/2023/CD2_test.txt')
    val_loader_cd2 = torch.utils.data.DataLoader(val_set_cd2, batch_size=batch_size, shuffle=True)
    val_set_d = MyDataset(txt_path=r'/home/Laizhm/proj/swin_ue/datasets/txt/2023/dfdc_test_lip.txt')
    val_loader_d = torch.utils.data.DataLoader(val_set_d, batch_size=batch_size, shuffle=True)
    val_set_cd1 = MyDataset(txt_path=r'/home/Laizhm/proj/swin_ue/datasets/txt/2023/CD1_test.txt')
    val_loader_cd1 = torch.utils.data.DataLoader(val_set_cd1, batch_size=batch_size, shuffle=True)
    val_set_dp = MyDataset(txt_path=r'/home/Laizhm/proj/swin_ue/datasets/txt/2023/dfdcp_test.txt')
    val_loader_dp = torch.utils.data.DataLoader(val_set_dp, batch_size=batch_size, shuffle=True)
    val_set_dfr = MyDataset(txt_path=r'/home/Laizhm/proj/swin_ue/datasets/txt/2023/DFR_test.txt')
    val_loader_dfr = torch.utils.data.DataLoader(val_set_dfr, batch_size=batch_size, shuffle=True)

    val_loaders = [val_loader_ffpp,val_loader_cd1,val_loader_cd2,val_loader_d,val_loader_dp,val_loader_dfr]
    val_names = ['ffpp','cd1','cd2','dfdc','dfdcp','dfr']

    net = MainModel(pretrained=False)
    net = net.cuda()

    with open(os.path.join(args.outf,"all_acc.txt"), "w") as f:
        for epoch in range(pre_epoch, EPOCH):
            path = args.outf + r'/net_%03d.pth' % (epoch+1)
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
    val_set_d = MyDataset(txt_path=r'/home/Laizhm/proj/swin_ue/datasets/txt/2023/FFIW_test.txt')
    val_loader_d = torch.utils.data.DataLoader(val_set_d, batch_size=64, shuffle=True)

    net = MainModel(pretrained=False)
    net = net.cuda()

    with open(os.path.join(args.outf,"ffiw_acc.txt"), "w+") as f:
        for epoch in range(0, EPOCH):
            path = args.outf + r'/net_%03d.pth' % (epoch+1)
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
                eer = 100. * compute_eer(labels, pre)
                print('%s|acc:%.3f%%, auc:%.3f%%, eer:%.3f%%' % ('ffiw',acc,auc,eer))
                f.write("%s|acc:%.3f%%, auc:%.3f%%, eer:%.3f%%; " % ('ffiw',acc,auc,eer))

                f.write('\n')
                f.flush()
        print("Testing Finished, TotalEPOCH=%d" % EPOCH)

if __name__ == "__main__":
        # test_all()
        test_videos()
        # test_eer()