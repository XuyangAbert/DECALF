# My Paper Title

This repository is the official implementation of [Deep Active Learning using Clustering-based Sampling for High-dimensional Image Classification]. 

## Requirements

Install Python 3.10 first to set up the environment.

To install requirements:

```setup
pip install -r requirements.txt
```

## Dependencies
1. numpy
2. scipy
3. torch
4. torchvision
5. scikit-learn
6. tqdm
7. ipdb==0.13.9
8. openml==0.12.2
9. faiss-gpu==1.7.2
10. toma==1.1.0
11. opencv-python==4.5.5.64
12. wilds==2.0.0

## Example running experiments
For the pytorch implementation, use the google colab with A100 GPU instance by running the `calfd_pytorch_demo.py` file. User can upload it as the notebook and run the code in google colab. 

User can change the variable `ALStrategy` to change the AL method. The label budget can be adjusted by adjusting the parameters below:

`initseed`: the size of initial label set for training the base learner. We set it as 100 for all AL methods in FashionMNIST, CIFAR-10, and SVHN datasets. For CIFAR-100 and TinyImageNet, we set it to be 500 and 1000 respectively.

`NUM_QUERY`: the label budget after the initial model has been trained.

`NUM_ROUNDS`: we set it as 1 to conduct single round query.

To change the dataset name, please adjust the variable `DATA_NAME` to select from the datasets below:
1. FashionMNIST
2. CIFAR10
3. CIFAR100
4. SVHN
5. TinyImageNet

For the statistical analysis using Nemenyi-post hoc test, you can simply run the `stats_analysis.py` file to automatically generate the critical distance plot. The `Statistical_Res_Acc.csv` and `Statistical_Res_F1.csv` are provided to make sure the CD diagram can be generated. 

## Results

Our model achieves the following performance on :

### [Image Classification on FashionMNIST](https://paperswithcode.com/dataset/fashion-mnist)

| Model name         | Top 1 Accuracy (20% labeled data) | F1 score (20% labeled data) |
| ------------------ |---------------- | -------------- |
| ResNet-18 + DECALF (Proposed) |     88.78%         |      88.50%       |

### [Image Classification on CIFAR-10](https://paperswithcode.com/dataset/cifar-10)

| Model name         | Top 1 Accuracy (20% labeled data) | F1 score (20% labeled data) |
| ------------------ |---------------- | -------------- |
| ResNet-18 + DECALF (Proposed) |     74.47%         |      74.28%       |

### [Image Classification on CIFAR-100](https://paperswithcode.com/dataset/cifar-100)

| Model name         | Top 1 Accuracy (20% labeled data) | F1 score (20% labeled data) |
| ------------------ |---------------- | -------------- |
| ResNet-18 + DECALF (Proposed) |     54.98%         |      54.79%       |

### [Image Classification on SVHN](https://paperswithcode.com/dataset/svhn)

| Model name         | Top 1 Accuracy (20% labeled data) | F1 score (20% labeled data) |
| ------------------ |---------------- | -------------- |
| ResNet-18 + DECALF (Proposed) |     90.23%         |      89.43%       |

### [Image Classification on TinyImageNet](https://paperswithcode.com/dataset/tiny-imagenet)

| Model name         | Top 1 Accuracy (20% labeled data) | F1 score (20% labeled data) |
| ------------------ |---------------- | -------------- |
| ResNet-18 + DECALF (Proposed) |     26.18%         |      15.73%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  CC-BY-4.0
