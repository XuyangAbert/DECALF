# My Paper Title

This repository is the official implementation of [To be added]. 

## Requirements

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
For the pytorch implementation, use the google colab with A100 GPU instance by running the `calfd_pytorch_demo.ipynb` file.

For users who prefer to use the tensorflow, the implementation of DeepALCS method for CIFAR-10, 100, and SVHN datasets can be found from the following files:

1. `calfd_tensorflow_1.ipynb`
2. `calfd_tensorflow_2.ipynb`
3. `calfd_tensorflow_3.ipynb`


## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository.
