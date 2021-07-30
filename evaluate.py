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
from tool import imutils

import scipy.io as io
import cv2

import model_RW as model_RW
import model_nonRW as model_nonRW
import basic_function as func
import dataset
import transform
from scipy.special import entr

from IPython.core import debugger

debug = debugger.Pdb ().set_trace

parserWarpper = func.MyArgumentParser ( inference=True )
parser = parserWarpper.get_parser ()
args = parser.parse_args ()

opt_manualSeed = 1000
print("Random Seed: ", opt_manualSeed)
np.random.seed ( opt_manualSeed )
random.seed ( opt_manualSeed )
torch.manual_seed ( opt_manualSeed )
torch.cuda.manual_seed_all ( opt_manualSeed )

# cudnn.enabled = True
cudnn.benchmark = True
cudnn.deterministic = False
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

if not os.path.exists ( args.save_path ) :
    os.makedirs ( args.save_path )

def net_process(args, model, image, mean, std=None, flip=False):
    input = torch.from_numpy(image.transpose((2, 0, 1))).float()
    if std is None:
        for t, m in zip(input, mean):
            t.sub_(m)
    else:
        for t, m, s in zip(input, mean, std):
            t.sub_(m).div_(s)
    input = input.unsqueeze(0).cuda()
    if flip:
        input = torch.cat([input, input.flip(3)], 0)
    with torch.no_grad():
        if args.model_type == 'RW':
            output, _, _, _ = model ( input )
        elif args.model_type == 'nonRW':
            output = model(input)
    _, _, h_i, w_i = input.shape
    _, _, h_o, w_o = output.shape
    if (h_o != h_i) or (w_o != w_i):
        output = F.interpolate(output, (h_i, w_i), mode='bilinear', align_corners=True)
    output = F.softmax(output, dim=1)
    if flip:
        output = (output[0] + output[1].flip(2)) / 2
    else:
        output = output[0]
    output = output.data.cpu().numpy()
    output = output.transpose(1, 2, 0)
    
    return output


def scale_process(args, model, image, classes, crop_h, crop_w, h, w, mean, std=None, stride_rate=2/3):
    ori_h, ori_w, _ = image.shape
    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=mean)
    new_h, new_w, _ = image.shape
    stride_h = int(np.ceil(crop_h*stride_rate))
    stride_w = int(np.ceil(crop_w*stride_rate))
    grid_h = int(np.ceil(float(new_h-crop_h)/stride_h) + 1)
    grid_w = int(np.ceil(float(new_w-crop_w)/stride_w) + 1)
    prediction_crop = np.zeros((new_h, new_w, classes), dtype=float)
    count_crop = np.zeros((new_h, new_w), dtype=float)
    #debug()
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            image_crop = image[s_h:e_h, s_w:e_w].copy()
            count_crop[s_h:e_h, s_w:e_w] += 1
            prediction_crop[s_h:e_h, s_w:e_w, :] += net_process(args, model, image_crop, mean, std)
    prediction_crop /= np.expand_dims(count_crop, 2)
    prediction_crop = prediction_crop[pad_h_half:pad_h_half+ori_h, pad_w_half:pad_w_half+ori_w]
    
    prediction = cv2.resize(prediction_crop, (w, h), interpolation=cv2.INTER_LINEAR)
       
    return prediction


def test(args, test_loader, model, classes, mean, std, base_size, crop_h, crop_w, scales):
    tbar = tqdm ( val_loader )
    confusion_meter = tnt.meter.ConfusionMeter ( args.numclasses, normalized=False )
    model.eval()
    for i, (input, gt, img_path) in enumerate(tbar):
        input1 = np.squeeze(input.numpy(), axis=0)
        image = np.transpose(input1, (1, 2, 0))
        h, w, _ = image.shape
        prediction = np.zeros((h, w, classes), dtype=float)
        for scale in scales:
            long_size = round(scale * base_size)
            new_h = long_size
            new_w = long_size
            if h > w:
                new_w = round(long_size/float(h)*w)
            else:
                new_h = round(long_size/float(w)*h)
            image_scale = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            prediction += scale_process(args, model, image_scale, classes, crop_h, crop_w, h, w, mean, std)
            
        prediction /= len(scales)
        
        #entropy = entr(prediction).sum(axis=2)/np.log(2)
        #img_name = img_path[0][img_path[0].rfind ( '/' ) + 1 :-4]
        #io.savemat(os.path.join ( args.save_path, img_name + '.mat' ), {'name': entropy})
        
        prediction = np.argmax(prediction, axis=2)
        pred=torch.from_numpy(prediction)
        pred=torch.unsqueeze(pred,0)
        valid_pixel = gt.ne(255)
        confusion_meter.add(pred[valid_pixel], gt[valid_pixel])
        
        mask = func.get_mask_pallete ( pred[0].cpu ().numpy (), 'pascal_voc' )

        img_name = img_path[0][img_path[0].rfind ( '/' ) + 1 :-4]
        mask.save ( os.path.join ( args.save_path, img_name + '.png' ) )
        #vutils.save_image ( input[0], os.path.join ( args.save_path, img_name + '.jpg' ), nrow=1, padding=0, normalize=True )
                                
    confusion_matrix = confusion_meter.value()
    inter = np.diag(confusion_matrix)
    union = confusion_matrix.sum(1).clip(min=1e-12) + confusion_matrix.sum(0).clip(min=1e-12) - inter

    mean_iou_ind = inter/union
    mean_iou_all = mean_iou_ind.mean()
    mean_acc_pix = float(inter.sum())/float(confusion_matrix.sum())
    print(' * IOU_All {iou}'.format(iou=mean_iou_all))
    print(' * IOU_Ind {iou}'.format(iou=mean_iou_ind))
    print(' * ACC_Pix {acc}'.format(acc=mean_acc_pix))

value_scale = 255
mean = [0.485, 0.456, 0.406]
mean = [item * value_scale for item in mean]
std = [0.229, 0.224, 0.225]
std = [item * value_scale for item in std]
val_transform = transform.Compose ( [transform.ToTensor ()])
val_dataset = dataset.SemData ( split='val', data_root=args.dataset_path, data_list='val.txt',
                                transform=val_transform, path='SegmentationClassAug' )
val_loader = data.DataLoader ( val_dataset, num_workers=args.workers,
                               batch_size=1, shuffle=False, pin_memory=True )

if args.model_type == 'RW':
    model = model_RW.Res_Deeplab ( num_classes=args.numclasses, layers=args.layers,
                                    shrink_factor= args.shrink_factor)
elif args.model_tpye == 'nonRW':
    model = model_nonRW.Res_Deeplab ( num_classes=args.numclasses, layers=args.layers )

model_pretrain = torch.load ( args.checkpoint_path )
model = func.param_restore_all ( model, model_pretrain['state_dict'] )
model = torch.nn.DataParallel ( model )
model = model.cuda ()

test(args, val_loader, model, args.numclasses, mean, std, 512, 465, 465, [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])

for item in args.__dict__.items():
    print(item)