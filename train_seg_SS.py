import os
import sys
import time
import shutil
import random
import numpy as np
import torchnet as tnt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.models as models
import torch.backends.cudnn as cudnn
from torch.autograd import Function
from torch.utils import data
import torch.distributed as dist
from datetime import datetime
from tqdm import tqdm
import socket

import model_RW as model_RW
import model_nonRW as model_nonRW
import basic_function as func
import dataset
import transform_contour as transform_contour
import transform as transform
import loss as loss

from IPython.core import debugger
debug = debugger.Pdb().set_trace

from tensorboardX import SummaryWriter
writer=SummaryWriter()

parserWarpper = func.MyArgumentParser()
parser = parserWarpper.get_parser()
args = parser.parse_args()
#print [item for item in args.__dict__.items()]

opt_manualSeed = 1000
print("Random Seed: ", opt_manualSeed)
np.random.seed(opt_manualSeed)
random.seed(opt_manualSeed)
torch.manual_seed(opt_manualSeed)
torch.cuda.manual_seed_all(opt_manualSeed)

#cudnn.enabled = True
cudnn.benchmark = True
cudnn.deterministic = False
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


class Trainer():
    def __init__(self, args):
        self.args = args
        self.date = datetime.now().strftime('%b%d_%H-%M-%S')+'_'+socket.gethostname()
        self.best_pred = 0
        
        
        value_scale = 255
        mean = [0.485, 0.456, 0.406]
        mean = [item * value_scale for item in mean]
        std = [0.229, 0.224, 0.225]
        std = [item * value_scale for item in std]

        train_transform = transform_contour.Compose([
            transform_contour.RandScale([0.5, 2.0]),
            transform_contour.RandRotate([-10, 10], padding=mean, ignore_label=255),
            transform_contour.RandomGaussianBlur(),
            transform_contour.RandomHorizontalFlip(),
            transform_contour.Crop([465, 465], crop_type='rand', padding=mean, ignore_label=255),
            transform_contour.ToTensor(),
            transform_contour.Normalize(mean=mean, std=std)])
        train_dataset = dataset.Sem_ContourData(split='train', data_root=args.dataset_path, data_list='train.txt', transform=train_transform, path = args.train_path, path_contour= 'superpixels')
        
        val_transform = transform.Compose([
            transform.Crop([465, 465], crop_type='center', padding=mean, ignore_label=255),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])
        val_dataset = dataset.SemData(split='val', data_root=args.dataset_path, data_list='val.txt', transform=val_transform, path = 'SegmentationClassAug')

        self.train_loader = data.DataLoader(train_dataset, num_workers=args.workers, batch_size=args.batchsize, shuffle=True, pin_memory=True)
        self.val_loader = data.DataLoader(val_dataset, num_workers=args.workers, batch_size=int(args.batchsize/2), shuffle=False, pin_memory=True)

        resnet = models.__dict__['resnet' + str(args.layers)](pretrained=True)
        self.model = model_RW.Res_Deeplab(num_classes=args.numclasses, layers=args.layers, shrink_factor=args.shrink_factor)

        if args.model_path != 'None':
            model_resnet = torch.load(args.model_path)
            self.model = func.param_restore_all(self.model, model_resnet['state_dict'])
        else:
            self.model.model_sed = func.param_restore(self.model.model_sed, resnet.state_dict())

        max_step = args.epochs * len(self.train_loader)
        
        self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),lr=args.lr, momentum=args.momentum, weight_decay=args.wdecay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: 10 * ((1.0-float(step)/max_step)**0.9))
        self.criterion_CE = func.SegLoss(255)
        self.criterion_entropy = func.EntropyLoss()
        
        self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.cuda()
        
        x=((465-1)//8)//args.shrink_factor+1
        y=((465-1)//8)//args.shrink_factor+1
        self.Trans = func.get_flip_transfer(x, y)
        self.criterion_selfsupervise_mse = loss.MSELoss_mask()
        self.criterion_selfsupervise_eig = loss.EigLoss()

    def train(self, epoch):
        self.model.train()
        losses = func.AverageMeter()
        sslosses = func.AverageMeter()
        tbar = tqdm(self.train_loader)
        for i, batch in enumerate(tbar):
            cur_lr = self.scheduler.get_lr()[0]
            img, gt, edge, path_name = batch
            
            batch_size = img.size()[0]
            input_size = img.size()[2:4]
            
            img_v = img.cuda(non_blocking=True)
            gt_v = gt.cuda(non_blocking=True)
            edge_v =edge.cuda(non_blocking=True)
            
            random_type = -1
            if self.args.selfloss_type=='random':
                random_type=random.randint(1,2)

            if self.args.selfloss_type == 'flip' or random_type == 1:
                img_flip_v = torch.flip(img_v,[3])
                T = torch.unsqueeze(self.Trans, 0).repeat(batch_size, 1, 1)
                T = T.cuda(non_blocking=True)
            elif self.args.selfloss_type == 'translation' or random_type == 2:
                crop_type=random.random()
                if crop_type<0.5:
                    img_flip_v = img_v[:,:,0:385,:].clone()
                else:
                    img_flip_v = img_v[:,:,80:465].clone()

            pred, P, x, z = self.model(img_v)
            with torch.no_grad():
                pred_s, P_s, x_s, z_s = self.model(img_flip_v)
                P_s.detach_()
                x_s.detach_()
                z_s.detach_()
            
            
            loss_self_supervised=0
            ss_weight=30
            if self.args.selfloss_feature == 'x':
                feature = x
                feature_s = x_s
            elif self.args.selfloss_feature == 'z':
                feature = z
                feature_s = z_s
            elif self.args.selfloss_feature == 'P':
                feature = P
                feature_s = P_s
            if self.args.selfloss_type == 'flip' or random_type == 1:
                if self.args.selfloss_feature == 'x' or self.args.selfloss_feature == 'z':
                    loss_self_supervised = self.criterion_selfsupervise_mse(feature, torch.flip(feature_s,[3]),None,0)
                elif self.args.selfloss_feature == 'P':
                    loss_self_supervised = self.criterion_selfsupervise_eig(feature, torch.bmm(torch.bmm(T, feature_s), T))
            elif self.args.selfloss_type == 'translation' or random_type == 2:
                if self.args.selfloss_feature == 'x' or self.args.selfloss_feature == 'z':
                    _, _, h_s, w_s = feature_s.size()
                    _, _, h, w = feature.size()
                    if crop_type < 0.5:
                        loss_self_supervised = self.criterion_selfsupervise_mse(feature[:,:,0:h_s,0:w_s], feature_s, None, 0)
                    else:
                        loss_self_supervised = self.criterion_selfsupervise_mse(feature[:,:,h-h_s:h,w-w_s:w], feature_s, None, 0)
                elif self.args.selfloss_feature == 'P':
                    _, h_s, w_s = feature_s.size()
                    _, h, w = feature.size()
                    if crop_type < 0.5:
                        loss_self_supervised = self.criterion_selfsupervise_eig(feature[:,0:h_s,0:w_s], feature_s)
                    else:
                        loss_self_supervised = self.criterion_selfsupervise_eig(feature[:,h-h_s:h,w-w_s:w], feature_s)

            loss_self_supervised = ss_weight*loss_self_supervised
            pred_sg_up = F.interpolate(pred, size=input_size, mode='bilinear', align_corners=True)
            loss = self.criterion_CE(pred_sg_up, gt_v.squeeze(1)) + loss_self_supervised
            loss_edge = 0

            if self.args.edgeloss_weight != 0:
                edge_v = edge.cuda(non_blocking=True)
                loss_edge=self.criterion_entropy(pred_sg_up,edge_v,args.use_boundary)
                loss = loss + self.args.edgeloss_weight * loss_edge

            losses.update(loss.item(), img.size(0))
            sslosses.update(30 * loss_self_supervised.item(), img.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            #self.scheduler.step()

            tbar.set_description('Train [{0}] Loss {loss.val:.3f} {loss.avg:.3f} SSLoss {SSloss.avg:.3f} Lr {lr:.5f} Best {best:.4f}'.format(epoch, loss=losses, SSloss=sslosses, lr=cur_lr, best=self.best_pred))
        writer.add_scalar('train/train_loss',losses.avg,epoch)


    def validate_tnt(self, epoch):
        confusion_meter = tnt.meter.ConfusionMeter(self.args.numclasses, normalized=False)
        losses = func.AverageMeter()
        tbar = tqdm(self.val_loader)
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tbar):
                img, gt,_ = batch

                batch_size = img.size()[0]
                input_size = img.size()[2:4]
                
                img_v = img.cuda(non_blocking=True)
                gt_v = gt.cuda(non_blocking=True)

                pred,_,_,_ = self.model(img_v)

                pred_sg_up = F.interpolate(pred, size=input_size, mode='bilinear', align_corners=True)

                loss = self.criterion_CE(pred_sg_up, gt_v.squeeze(1))
                
                valid_pixel = gt.ne(255)
                pred_sg_up_label = torch.max(pred_sg_up, 1, keepdim=True)[1]
                pred_sg_up_label=torch.squeeze(pred_sg_up_label,1)
                
                confusion_meter.add(pred_sg_up_label[valid_pixel], gt[valid_pixel])
                losses.update(loss.item(), img.size(0))
                tbar.set_description('Valid [{0}] Loss {loss.val:.3f} {loss.avg:.3f}'.format(epoch, loss=losses))
                if i == 0:
                    colormap = func.vocpallete
                    cp=torch.from_numpy(np.array(colormap)).reshape((-1,3)).float()
                    pred=cp[pred_sg_up_label,:].squeeze().permute(0,3,1,2)
                    label=cp[gt,:].squeeze().permute(0,3,1,2)
                    imgshow = torch.cat((label,pred),0)
                    img_grid = vutils.make_grid(imgshow,nrow=batch_size)
                    writer.add_image('gt&pred',img_grid,epoch)


            confusion_matrix = confusion_meter.value()
            inter = np.diag(confusion_matrix)
            union = confusion_matrix.sum(1).clip(min=1e-12) + confusion_matrix.sum(0).clip(min=1e-12) - inter

            mean_iou_ind = inter/union
            mean_iou_all = mean_iou_ind.mean()
            mean_acc_pix = float(inter.sum())/float(confusion_matrix.sum())
            print(' * IOU_All {iou}'.format(iou=mean_iou_all))
            print(' * IOU_Ind {iou}'.format(iou=mean_iou_ind))
            print(' * ACC_Pix {acc}'.format(acc=mean_acc_pix))
            writer.add_scalar('val/val_loss',losses.avg,epoch)
            writer.add_scalar('val/val_iou',mean_iou_all,epoch)

        return mean_iou_all, mean_iou_ind, mean_acc_pix

trainer = Trainer(args)
#trainer.validate_tnt(0)
for epoch in range(args.epochs):

    # train and validate
    trainer.train(epoch)
    iou_all, iou_ind, acc_pix = trainer.validate_tnt(epoch)

    # save checkpoint
    is_best = iou_all > trainer.best_pred
    trainer.best_pred = iou_all if is_best else trainer.best_pred

    func.save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': trainer.model.state_dict(),
        'best_pred': (iou_all, iou_ind, acc_pix),
        'optimizer': trainer.optimizer.state_dict(),
    }, trainer.date, is_best, trainer.args.shfilename)

        
writer.close()
