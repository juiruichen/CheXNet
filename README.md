# Intelligent Thoracic Disease Classification and Interpretation Assistance System

## Dataset

The [ChestX-ray14 dataset](http://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf) comprises 112,120 frontal-view chest X-ray images of 30,805 unique patients with 14 disease labels. To evaluate the model, we randomly split the dataset into training (70%), validation (10%) and test (20%) sets, following the work in paper. Partitioned image names and corresponding labels are placed under the directory [labels](./ChestX-ray14/labels).

## Usage

1. Clone this repository.

2. Download images of ChestX-ray14 from this [released page](https://nihcc.app.box.com/v/ChestXray-NIHCC) and decompress them to the directory [images](./ChestX-ray14/images).

3. Specify one or multiple GPUs.

4. Run training command: `python main.py`.

5. Run inference command: `python inference.py`.

## Comparsion

We followed the training strategy described in the official paper, and a ten crop method is adopted both in validation and test. Compared with the original CheXNet, the per-class AUROC of our reproduced model is almost the same.

|     Pathology      | [Wang et al.](https://arxiv.org/abs/1705.02315) | [Yao et al.](https://arxiv.org/abs/1710.10501) | [CheXNet](https://arxiv.org/abs/1711.05225) | Our Implemented CheXNet |
| :----------------: | :--------------------------------------: | :--------------------------------------: | :--------------------------------------: | :---------------------: |
|    Atelectasis     |                  0.716                   |                  0.772                   |                  0.8094                  |         0.8123          |
|    Cardiomegaly    |                  0.807                   |                  0.904                   |                  0.9248                  |         0.9132          |
|      Effusion      |                  0.784                   |                  0.859                   |                  0.8638                  |         0.8826          |
|    Infiltration    |                  0.609                   |                  0.695                   |                  0.7345                  |         0.7099          |
|        Mass        |                  0.706                   |                  0.792                   |                  0.8676                  |         0.8552          |
|       Nodule       |                  0.671                   |                  0.717                   |                  0.7802                  |         0.7609          |
|     Pneumonia      |                  0.633                   |                  0.713                   |                  0.7680                  |         0.7708          |
|    Pneumothorax    |                  0.806                   |                  0.841                   |                  0.8887                  |         0.8731          |
|   Consolidation    |                  0.708                   |                  0.788                   |                  0.7901                  |         0.8101          |
|       Edema        |                  0.835                   |                  0.882                   |                  0.8878                  |         0.8952          |
|     Emphysema      |                  0.815                   |                  0.829                   |                  0.9371                  |         0.9115          |
|      Fibrosis      |                  0.769                   |                  0.767                   |                  0.8047                  |         0.8356          |
| Pleural Thickening |                  0.708                   |                  0.765                   |                  0.8062                  |         0.7784          |
|       Hernia       |                  0.767                   |                  0.914                   |                  0.9164                  |         0.9447          |
|    Average AUROC   |                                          |                                          |                  0.8410                  |         0.8400          |

## Used models

### CNNs
- DenseNet-201 (densenet201)
- EfficientNet (efficientnet)
- EfficientNet V2 (efficientnetv2)

### Transformers
- ViT (vit)
- DeiT (deit)
- Swin Transformer (swin)
- Swin Transformer V2 (swinv2)

### Models Comparison
|     Pathology      |    DenseNet-201    |    EfficientNet    |         ViT        |        DeiT        |        Swin        |       Swin V2      |
| :----------------: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: |
|    Atelectasis     |       0.8123       |       0.8096       |||
|    Cardiomegaly    |       0.9132       |       0.9110       |||
|      Effusion      |       0.8826       |       0.8813       |||
|    Infiltration    |       0.7099       |       0.7062       |||
|        Mass        |       0.8552       |       0.8393       |||
|       Nodule       |       0.7609       |       0.7565       |||
|     Pneumonia      |       0.7708       |       0.7591       |||
|    Pneumothorax    |       0.8731       |       0.8698       |||
|   Consolidation    |       0.8101       |       0.8122       |||
|       Edema        |       0.8952       |       0.8943       |||
|     Emphysema      |       0.9115       |       0.8946       |||
|      Fibrosis      |       0.8356       |       0.8264       |||
| Pleural Thickening |       0.7784       |       0.7644       |||
|       Hernia       |       0.9447       |       0.9057       |||
|  **Average AUROC** |     **0.8400**     |     **0.8310**     |||

## Used Devices
 - CPU: 13th Gen Intel(R) Core(TM) i5-13500
 - GPU: NVIDIA GeForce RTX 4090
