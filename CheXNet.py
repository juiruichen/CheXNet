import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as transforms

from sklearn.metrics import roc_auc_score

from ChestXrayDataset import ChestXrayDataset


from models import densenet
from models import efficientnet

class CheXNet():
    
    def __init__(self,
                 model_name,
                 image_dir,
                 train_labels_path,
                 val_labels_path,
                 test_labels_path,
                 num_classes,
                 num_epochs,
                 batch_size,
                 pretrained,
                 learning_rate,
                 num_workers,
                 ):
        self.model_name = model_name
        self.image_dir = image_dir
        self.train_labels_path = train_labels_path
        self.val_labels_path = val_labels_path
        self.test_labels_path = test_labels_path
        self.batch_size = batch_size
        self.pretrained = pretrained
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if model_name == 'densenet201': 
            self.model = densenet.DenseNet201(num_classes).to(device)
        elif model_name == 'efficientnet':
            self.model = efficientnet.EfficientNet(num_classes).to(device)

        
        
    def train(self, checkpoint_dir: str=None):
        print("=> Start Training")
        writer = SummaryWriter(f"./runs/{self.model_name}/")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # TODO: For multiple GPUs
        # self.model = torch.nn.DataParallel(self.model).to(device)
        
        # TODO: Data preparing
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        dataset_train = ChestXrayDataset(data_dir=self.image_dir,
                                         image_list_file=self.train_labels_path,
                                         transform=transform)
        dataset_val = ChestXrayDataset(data_dir=self.image_dir,
                                         image_list_file=self.val_labels_path,
                                         transform=transform)
        
        dataloader_train = DataLoader(dataset=dataset_train,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      num_workers=self.num_workers,
                                      pin_memory=True)
        dataloader_val = DataLoader(dataset=dataset_val,
                                    batch_size=self.batch_size,
                                    shuffle=False,
                                    num_workers=self.num_workers,
                                    pin_memory=True)
        
        # TODO: Optimizer, Scheduler, Loss
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min')
        loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
        
        # TODO: Train and Validation
        loss_min = 100000
        for epoch in range(self.num_epochs):
            print("epoch", epoch+1)
            self.epoch_train(model=self.model, dataloader=dataloader_train, optimizer=optimizer, loss=loss, device=device)
            loss_val, losstensor = self.epoch_val(model=self.model, dataloader=dataloader_val, loss=loss, device=device)
            writer.add_scalar('Loss/Validation', loss_val, epoch)
            scheduler.step(losstensor.item())
            
            if loss_val < loss_min:
                loss_min = loss_val
                torch.save({"epoch": epoch+1, "state_dict": self.model.state_dict(), "best_loss": loss_min, "optimizer":optimizer.state_dict()}, checkpoint_dir+f'best_model_{self.model_name}.pth.tar')
                print ('Epoch [' + str(epoch + 1) + '] [save] [' + '] loss= ' + str(loss_val))
            else:
                print ('Epoch [' + str(epoch + 1) + '] [----] [' + '] loss= ' + str(loss_val))           
        
        print("=> End training")
        writer.flush()
        writer.close()
        
    def epoch_train(self, model, dataloader, optimizer, loss, device):
        model.train()
        
        for batch, (input, target) in enumerate(dataloader):
            # print(f"Batch number: {batch}")
            input = input.to(device)
            target = target.to(device)
            output = model(input)
            
            loss_value = loss(output, target)
            
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            
    def epoch_val(self, model, dataloader, loss, device):
        model.eval()
        
        lossVal = 0
        lossValNorm = 0
        losstensorMean = 0
        
        with torch.no_grad():
            for batch, (input, target) in enumerate (dataloader):
                target = target.to(device)     
                # varInput = torch.autograd.Variable(input)
                # varTarget = torch.autograd.Variable(target)  
                varInput = input.to(device)
                  
                varOutput = model(varInput)
                losstensor = loss(varOutput, target)
                losstensorMean += losstensor
                
                lossVal += losstensor.item()
                lossValNorm += 1
            
        outLoss = lossVal / lossValNorm
        losstensorMean = losstensorMean / lossValNorm
        
        return outLoss, losstensorMean
    
    def compute_AUCs(self, gt, pred):
        AUROCs = []
        gt_np = gt.cpu().numpy()
        pred_np = pred.cpu().numpy()
        for i in range(14):
            AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
        return AUROCs