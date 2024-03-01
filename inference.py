# encoding: utf-8

"""
The main CheXNet model implementation.
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import utils
from ChestXrayDataset import ChestXrayDataset
from models import densenet
from models import efficientnet

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default="./theta.json")
args = parser.parse_args()
params = utils.Params(args.config_path)
model_name = params.model_name
image_dir = params.data_dir
checkpoint_dir = params.checkpoint_dir
num_classes = params.num_classes
batch_size = params.batch_size
test_labels_path = params.test_file_path
class_names = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

def main():
    
    cudnn.benchmark = True

    # initialize and load the model
    if params.model_name=="densenet201": 
        model = densenet.DenseNet201(num_classes).cuda()
    elif params.model_name=="efficientnet":
        model=efficientnet.EfficientNet(num_classes).cuda()
    # model = torch.nn.DataParallel(model).cuda()

    if os.path.isfile(checkpoint_dir+f"/best_model_{model_name}.pth.tar"):
        print("=> loading checkpoint")
        checkpoint = torch.load(checkpoint_dir+f"/best_model_{model_name}.pth.tar")
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint")
    else:
        print("=> no checkpoint found")
        exit()

    test_dataset = ChestXrayDataset(data_dir=image_dir,
                                    image_list_file=test_labels_path,
                                    transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.TenCrop(224),
                                        transforms.Lambda
                                        (crops_to_tensors),
                                        transforms.Lambda
                                        (normalize_crops)
                                    ]))
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=8, pin_memory=True)

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor().cuda()
    pred = torch.FloatTensor().cuda()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, (inp, target) in enumerate(test_loader):
            target = target.cuda()
            gt = torch.cat((gt, target), 0)
            bs, n_crops, c, h, w = inp.size()
            input_var = torch.autograd.Variable(inp.view(-1, c, h, w).cuda())
            output = model(input_var)
            output_mean = output.view(bs, n_crops, -1).mean(1)
            pred = torch.cat((pred, output_mean.data), 0)

    AUROCs = compute_AUCs(gt, pred)
    AUROC_avg = np.array(AUROCs).mean()
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(num_classes):
        print('The AUROC of {} is {}'.format(class_names[i], AUROCs[i]))


def compute_AUCs(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores.

    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.

    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(num_classes):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs

def crops_to_tensors(crops):
    to_tensor = transforms.ToTensor()
    return torch.stack([to_tensor(crop) for crop in crops])

def normalize_crops(crops):
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
    return torch.stack([normalize(crop)for crop in crops])

if __name__ == '__main__':
    main()