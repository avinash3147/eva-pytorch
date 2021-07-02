# Assignemnt 7 (Late Assignment but on time)
## Problem statement:-
1. your colab file must
    * train resnet18 for 20 epochs on the CIFAR10 dataset
    * show loss curves for test and train datasets
    * show a gallery of 10 misclassified images
    * show gradcam output on 10 misclassified images.
2. EARLY SUBMISSIONS:
    * Train for 20 Epochs
    * 10 misclassified images
    * 10  GradCam output on ANY  misclassified images
    * Apply these transforms while training:
        * RandomCrop(32, padding=4)
        * CutOut(16x16)
    * achieve 87% accuracy, as many epochs as you want. Total Params to be less than 100k. 

## Folder structure:-
* Experiments 
  * 2 experiment notebooks are kept under the main folder.  
* Results 
  * Contains text files which has logs and summary for the model used 
  * loss and accuracy graphs 
* models 
  * contains the model design 
* utils 
  * contains all utility methods needed for training and validating model 
* main.py 
  * Main file which calls the required methods sequentially just like colab notebook

## Enchancements Added:-

* We have not used maxpooling a any time. 
* For the Transition block 1 only 1x1 conv s used to reduce the number of params.
* For the Transition block 2 we have used **Dilation Convolution** to decrease the size of image and get a higher receptive field output
* For the Transition block 3 we have used normal convolutions with stride as 2. 
* We have used **6 Depth wise Seperable convolutions** . 
* Use Cross Entropy for loss calculation 
* Data Augumentations using Albumenation Library. 

## Data Augumentation:-

The data augumentation techniques used are:-
* RandomCrop(32, padding=4)
* CutOut(16x16)


## Best Model:-

### Model Summary:-
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
       BatchNorm2d-2           [-1, 64, 32, 32]             128
            Conv2d-3           [-1, 64, 32, 32]          36,864
       BatchNorm2d-4           [-1, 64, 32, 32]             128
            Conv2d-5           [-1, 64, 32, 32]          36,864
       BatchNorm2d-6           [-1, 64, 32, 32]             128
        BasicBlock-7           [-1, 64, 32, 32]               0
            Conv2d-8           [-1, 64, 32, 32]          36,864
       BatchNorm2d-9           [-1, 64, 32, 32]             128
           Conv2d-10           [-1, 64, 32, 32]          36,864
      BatchNorm2d-11           [-1, 64, 32, 32]             128
       BasicBlock-12           [-1, 64, 32, 32]               0
           Conv2d-13          [-1, 128, 16, 16]          73,728
      BatchNorm2d-14          [-1, 128, 16, 16]             256
           Conv2d-15          [-1, 128, 16, 16]         147,456
      BatchNorm2d-16          [-1, 128, 16, 16]             256
           Conv2d-17          [-1, 128, 16, 16]           8,192
      BatchNorm2d-18          [-1, 128, 16, 16]             256
       BasicBlock-19          [-1, 128, 16, 16]               0
           Conv2d-20          [-1, 128, 16, 16]         147,456
      BatchNorm2d-21          [-1, 128, 16, 16]             256
           Conv2d-22          [-1, 128, 16, 16]         147,456
      BatchNorm2d-23          [-1, 128, 16, 16]             256
       BasicBlock-24          [-1, 128, 16, 16]               0
           Conv2d-25            [-1, 256, 8, 8]         294,912
      BatchNorm2d-26            [-1, 256, 8, 8]             512
           Conv2d-27            [-1, 256, 8, 8]         589,824
      BatchNorm2d-28            [-1, 256, 8, 8]             512
           Conv2d-29            [-1, 256, 8, 8]          32,768
      BatchNorm2d-30            [-1, 256, 8, 8]             512
       BasicBlock-31            [-1, 256, 8, 8]               0
           Conv2d-32            [-1, 256, 8, 8]         589,824
      BatchNorm2d-33            [-1, 256, 8, 8]             512
           Conv2d-34            [-1, 256, 8, 8]         589,824
      BatchNorm2d-35            [-1, 256, 8, 8]             512
       BasicBlock-36            [-1, 256, 8, 8]               0
           Conv2d-37            [-1, 512, 4, 4]       1,179,648
      BatchNorm2d-38            [-1, 512, 4, 4]           1,024
           Conv2d-39            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-40            [-1, 512, 4, 4]           1,024
           Conv2d-41            [-1, 512, 4, 4]         131,072
      BatchNorm2d-42            [-1, 512, 4, 4]           1,024
       BasicBlock-43            [-1, 512, 4, 4]               0
           Conv2d-44            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-45            [-1, 512, 4, 4]           1,024
           Conv2d-46            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-47            [-1, 512, 4, 4]           1,024
       BasicBlock-48            [-1, 512, 4, 4]               0
           Linear-49                   [-1, 10]           5,130
================================================================
Total params: 11,173,962
Trainable params: 11,173,962
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 11.25
Params size (MB): 42.63
Estimated Total Size (MB): 53.89
----------------------------------------------------------------

### Training Logs:-
0%|          | 0/391 [00:00<?, ?it/s]Epoch 1:
Loss=1.3807648420333862 Batch_id=390 Accuracy=37.83: 100%|██████████| 391/391 [01:04<00:00,  6.03it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0099, Accuracy: 27478/50000 (54.96%)

Epoch 2:
Loss=1.2307568788528442 Batch_id=390 Accuracy=52.55: 100%|██████████| 391/391 [01:04<00:00,  6.05it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0091, Accuracy: 30665/50000 (61.33%)

Epoch 3:
Loss=1.1359974145889282 Batch_id=390 Accuracy=59.25: 100%|██████████| 391/391 [01:05<00:00,  5.97it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0074, Accuracy: 33381/50000 (66.76%)

Epoch 4:
Loss=0.8834953308105469 Batch_id=390 Accuracy=64.96: 100%|██████████| 391/391 [01:05<00:00,  5.94it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0067, Accuracy: 35588/50000 (71.18%)

Epoch 5:
Loss=0.846274733543396 Batch_id=390 Accuracy=69.05: 100%|██████████| 391/391 [01:05<00:00,  5.95it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0050, Accuracy: 39021/50000 (78.04%)

Epoch 6:
Loss=0.7453653812408447 Batch_id=390 Accuracy=72.04: 100%|██████████| 391/391 [01:05<00:00,  5.97it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0059, Accuracy: 37485/50000 (74.97%)

Epoch 7:
Loss=0.4584648013114929 Batch_id=390 Accuracy=74.70: 100%|██████████| 391/391 [01:05<00:00,  5.98it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0041, Accuracy: 41081/50000 (82.16%)

Epoch 8:
Loss=0.6761376261711121 Batch_id=390 Accuracy=76.38: 100%|██████████| 391/391 [01:05<00:00,  5.96it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0037, Accuracy: 41987/50000 (83.97%)

Epoch 9:
Loss=0.46041959524154663 Batch_id=390 Accuracy=78.02: 100%|██████████| 391/391 [01:05<00:00,  5.93it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0030, Accuracy: 43357/50000 (86.71%)

Epoch 10:
Loss=0.8709599375724792 Batch_id=390 Accuracy=79.31: 100%|██████████| 391/391 [01:05<00:00,  5.96it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0024, Accuracy: 44657/50000 (89.31%)

Epoch 11:
Loss=0.5557038187980652 Batch_id=390 Accuracy=80.51: 100%|██████████| 391/391 [01:05<00:00,  5.97it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0024, Accuracy: 44658/50000 (89.32%)

Epoch 12:
Loss=0.4747753143310547 Batch_id=390 Accuracy=81.60: 100%|██████████| 391/391 [01:05<00:00,  5.98it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0020, Accuracy: 45678/50000 (91.36%)

Epoch 13:
Loss=0.49158453941345215 Batch_id=390 Accuracy=82.87: 100%|██████████| 391/391 [01:05<00:00,  5.96it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0018, Accuracy: 46076/50000 (92.15%)

Epoch 14:
Loss=0.4296746253967285 Batch_id=390 Accuracy=84.11: 100%|██████████| 391/391 [01:05<00:00,  5.95it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0014, Accuracy: 47022/50000 (94.04%)

Epoch 15:
Loss=0.3189294934272766 Batch_id=390 Accuracy=85.04: 100%|██████████| 391/391 [01:05<00:00,  5.98it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0013, Accuracy: 47053/50000 (94.11%)

Epoch 16:
Loss=0.4269600808620453 Batch_id=390 Accuracy=86.27: 100%|██████████| 391/391 [01:05<00:00,  5.98it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0009, Accuracy: 48017/50000 (96.03%)

Epoch 17:
Loss=0.36634740233421326 Batch_id=390 Accuracy=87.64: 100%|██████████| 391/391 [01:05<00:00,  5.97it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0007, Accuracy: 48536/50000 (97.07%)

Epoch 18:
Loss=0.48459696769714355 Batch_id=390 Accuracy=88.61: 100%|██████████| 391/391 [01:05<00:00,  5.96it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0006, Accuracy: 48977/50000 (97.95%)

Epoch 19:
Loss=0.4799357056617737 Batch_id=390 Accuracy=89.47: 100%|██████████| 391/391 [01:05<00:00,  5.97it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0005, Accuracy: 49135/50000 (98.27%)

Epoch 20:
Loss=0.36189454793930054 Batch_id=390 Accuracy=89.84: 100%|██████████| 391/391 [01:05<00:00,  5.96it/s]

Test set: Average loss: 0.0005, Accuracy: 49156/50000 (98.31%)

### Goals Achieved:-
* Epochs - 20 
* Total Params - **11,173,962 (Less than 100K)**
* Best Training Accuracy - **89.84%**
* Best Testing Accuracy - **98.31%**

### Accuracy of each class:-
![image](https://user-images.githubusercontent.com/51078583/122612990-b3377400-d0a1-11eb-87af-ef8065aac510.png)

### Validation Loss Curve:-
![image](https://user-images.githubusercontent.com/51078583/122613662-dadb0c00-d0a2-11eb-96d2-43d132733913.png)


## References:-

## Contributors:-
1. Avinash Ravi
2. Nandam Sriranga Chaitanya
3. Saroj Raj Das
4. Ujjwal Gupta
