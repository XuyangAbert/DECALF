import numpy as np
import torch
import random
import os
from torchvision import datasets
from PIL import Image
import requests
import zipfile
from collections import Counter
from sklearn.metrics import f1_score, confusion_matrix

class Data:
    def __init__(self, X_train, Y_train, X_test, Y_test, handler, args_task):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.handler = handler
        self.args_task = args_task
        
        self.n_pool = len(X_train)
        self.n_test = len(X_test)
        
        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)
        
    def initialize_labels(self, num):
        # generate initial labeled pool
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        self.labeled_idxs[tmp_idxs[:num]] = True
    
    def get_unlabeled_data_by_idx(self, idx):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return self.X_train[unlabeled_idxs][idx]
    
    def get_data_by_idx(self, idx):
        return self.X_train[idx], self.Y_train[idx]

    def get_new_data(self, X, Y):
        return self.handler(X, Y, self.args_task['transform_train'])

    def get_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return labeled_idxs, self.handler(self.X_train[labeled_idxs], self.Y_train[labeled_idxs], self.args_task['transform_train'])
    
    def get_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, self.handler(self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs], self.args_task['transform_train'])
    
    def get_train_data(self):
        return self.labeled_idxs.copy(), self.handler(self.X_train, self.Y_train, self.args_task['transform_train'])

    def get_test_data(self):
        return self.handler(self.X_test, self.Y_test, self.args_task['transform'])
    
    def get_partial_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return self.X_train[labeled_idxs], self.Y_train[labeled_idxs]
    
    def get_partial_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs]

    def cal_test_acc(self, preds):
        return 1.0 * (self.Y_test==preds).sum().item() / self.n_test

    def cal_test_f1(self, preds):
        return f1_score(self.Y_test, preds, average='macro')

    def cal_classwise_metrics(self, preds):
        cm = confusion_matrix(self.Y_test, preds)
        precision = []
        recall = []
        
        # Calculate precision and recall for each class
        for i in range(len(cm)):
            tp = cm[i, i]  # True positives for class i
            fp = cm[:, i].sum() - tp  # False positives for class i
            fn = cm[i, :].sum() - tp  # False negatives for class i
            
            # Precision: TP / (TP + FP)
            precision_i = tp / (tp + fp) if (tp + fp) > 0 else 0
            # Recall: TP / (TP + FN)
            recall_i = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            precision.append(precision_i)
            recall.append(recall_i)
        
        return precision, recall

def download_and_unzip(url, extract_to='.'):
    """
    Download a zip file from a URL and unzip it in the specified directory.

    Parameters:
        url (str): URL to the zip file.
        extract_to (str): Directory path where the contents will be extracted.
    """
    local_zip_path = os.path.join(extract_to, 'tiny-imagenet.zip')

    # Downloading the file by streaming the response
    response = requests.get(url, stream=True)
    with open(local_zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=128):
            f.write(chunk)
    # Extracting the zip file
    with zipfile.ZipFile(local_zip_path, 'r') as z:
        z.extractall(path=extract_to)
    os.remove(local_zip_path)
    print("Downloaded and extracted Tiny ImageNet")


def get_MNIST(handler, args_task):
    raw_train = datasets.MNIST('./data/MNIST', train=True, download=True)
    raw_test = datasets.MNIST('./data/MNIST', train=False, download=True)
    return Data(raw_train.data, raw_train.targets, raw_test.data, raw_test.targets, handler, args_task)

def get_MNIST_imb(handler, args_task):
    raw_train = datasets.MNIST('./data/MNIST', train=True, download=True)
    raw_test = datasets.MNIST('./data/MNIST', train=False, download=True)
    X_tr = raw_train.data
    Y_tr = torch.from_numpy(np.array(data_train.targets)).long()
    X_te = raw_test.data
    Y_te = torch.from_numpy(np.array(data_test.targets)).long()
    ratio = [0.4, 0.4, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    X_tr_imb = []
    Y_tr_imb = []
    random.seed(4666)
    for i in range(Y_tr.shape[0]):
        tmp = random.random()
        if tmp < ratio[Y_tr[i]]:
            X_tr_imb.append(X_tr[i])
            Y_tr_imb.append(Y_tr[i])
    X_tr_imb = np.array(X_tr_imb)
    Y_tr_imb = torch.LongTensor(np.array(Y_tr_imb)).type_as(Y_tr)
    return Data(X_tr_imb, Y_tr_imb, X_te, Y_te, handler, args_task)


def get_FashionMNIST(handler, args_task):
    raw_train = datasets.FashionMNIST('./data/FashionMNIST', train=True, download=True)
    raw_test = datasets.FashionMNIST('./data/FashionMNIST', train=False, download=True)
    return Data(raw_train.data, raw_train.targets, raw_test.data, raw_test.targets, handler, args_task)

def get_FashionMNIST_imb(handler, args_task):
    raw_train = datasets.FashionMNIST('./data/FashionMNIST', train=True, download=True)
    raw_test = datasets.FashionMNIST('./data/FashionMNIST', train=False, download=True)
    X_tr = raw_train.data
    Y_tr = torch.from_numpy(np.array(raw_train.targets)).long()
    X_te = raw_test.data
    Y_te = torch.from_numpy(np.array(raw_test.targets)).long()
    ratio = [0.4, 0.4, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    X_tr_imb = []
    Y_tr_imb = []
    random.seed(4666)
    for i in range(Y_tr.shape[0]):
        tmp = random.random()
        if tmp < ratio[Y_tr[i]]:
            X_tr_imb.append(X_tr[i])
            Y_tr_imb.append(Y_tr[i])
    X_tr_imb = np.array(X_tr_imb)
    Y_tr_imb = torch.LongTensor(np.array(Y_tr_imb)).type_as(Y_tr)
    return Data(X_tr_imb, Y_tr_imb, X_te, Y_te, handler, args_task)


def get_EMNIST(handler, args_task):
    raw_train = datasets.EMNIST('./data/EMNIST', split = 'byclass', train=True, download=True)
    raw_test = datasets.EMNIST('./data/EMNIST', split = 'byclass', train=False, download=True)
    return Data(raw_train.data, raw_train.targets, raw_test.data, raw_test.targets, handler, args_task)

def get_SVHN(handler, args_task):
    data_train = datasets.SVHN('./data/SVHN', split='train', download=True)
    data_test = datasets.SVHN('./data/SVHN', split='test', download=True)
    return Data(data_train.data, torch.from_numpy(data_train.labels), data_test.data, torch.from_numpy(data_test.labels), handler, args_task)

def get_CIFAR10(handler, args_task):
    data_train = datasets.CIFAR10('./data/CIFAR10', train=True, download=True)
    data_test = datasets.CIFAR10('./data/CIFAR10', train=False, download=True)
    return Data(data_train.data, torch.LongTensor(data_train.targets), data_test.data, torch.LongTensor(data_test.targets), handler, args_task)

def get_CIFAR10_imb(handler, args_task):
    data_train = datasets.CIFAR10('./data/CIFAR10', train=True, download=True)
    data_test = datasets.CIFAR10('./data/CIFAR10', train=False, download=True)
    X_tr = data_train.data
    Y_tr = torch.from_numpy(np.array(data_train.targets)).long()
    X_te = data_test.data
    Y_te = torch.from_numpy(np.array(data_test.targets)).long()
    ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    X_tr_imb = []
    Y_tr_imb = []
    random.seed(4666)
    for i in range(Y_tr.shape[0]):
        tmp = random.random()
        if tmp < ratio[Y_tr[i]]:
            X_tr_imb.append(X_tr[i])
            Y_tr_imb.append(Y_tr[i])
    X_tr_imb = np.array(X_tr_imb).astype(X_tr.dtype)
    Y_tr_imb = torch.LongTensor(np.array(Y_tr_imb)).type_as(Y_tr)
    return Data(X_tr_imb, Y_tr_imb, X_te, Y_te, handler, args_task)

def get_CIFAR100(handler, args_task):
    data_train = datasets.CIFAR100('./data/CIFAR100', train=True, download=True)
    data_test = datasets.CIFAR100('./data/CIFAR100', train=False, download=True)
    return Data(data_train.data, torch.LongTensor(data_train.targets), data_test.data, torch.LongTensor(data_test.targets), handler, args_task)

def get_CIFAR100_imbalanced(handler, args_task):
    # Load the CIFAR-100 dataset
    data_train = datasets.CIFAR100('./data/CIFAR100', train=True, download=True)
    data_test = datasets.CIFAR100('./data/CIFAR100', train=False, download=True)

    # Split the training data by class
    class_data = {i: [] for i in range(100)}
    for img, label in zip(data_train.data, data_train.targets):
        class_data[label].append((img, label))

    # Create an imbalanced dataset by reducing samples for some classes
    imbalanced_train_data = []
    imbalance_ratio = 0.1
    for class_idx, samples in class_data.items():
        if class_idx % 2 == 0:
            # For even classes, reduce the number of samples according to the imbalance ratio
            num_samples_to_retain = int(len(samples) * imbalance_ratio)
            imbalanced_train_data.extend(random.sample(samples, num_samples_to_retain))
        else:
            # For odd classes, keep all samples
            imbalanced_train_data.extend(samples)

    # Shuffle the imbalanced training data
    random.shuffle(imbalanced_train_data)

    # Extract the data and labels separately
    imbalanced_train_images = np.array([img for img, _ in imbalanced_train_data])
    imbalanced_train_labels = np.array([label for _, label in imbalanced_train_data])

    # # Check the new class distribution
    # class_counts = Counter(imbalanced_train_labels)
    # print("Class distribution in the imbalanced training dataset:", class_counts)

    # Return the Data object using the imbalanced training data
    return Data(
        imbalanced_train_images,
        torch.LongTensor(imbalanced_train_labels),
        data_test.data,
        torch.LongTensor(data_test.targets),
        handler,
        args_task
    )
def get_CIFAR100_overlapping_classes(handler, args_task, overlapping_classes=None):
    """
    Load the CIFAR-100 dataset and filter it to include only the specified overlapping classes.
    
    Args:
        handler: Data handler for further processing.
        args_task: Additional arguments for the task.
        overlapping_classes (list): List of class names to include in the experiment.
    
    Returns:
        Filtered CIFAR-100 dataset containing only the overlapping classes.
    """
    # Default overlapping classes if not provided
    if overlapping_classes is None:
        overlapping_classes = ["forest", "plain", "lion", "tiger", "cloud", "mountain", "bed", "couch", "apple", "pear"]

    # Load the CIFAR-100 dataset with fine labels (i.e., the actual class names)
    data_train = datasets.CIFAR100('./data/CIFAR100', train=True, download=True)
    data_test = datasets.CIFAR100('./data/CIFAR100', train=False, download=True)

    # Get the class to index mapping
    class_to_idx = data_train.class_to_idx

    # Find the indices corresponding to the overlapping classes
    overlapping_indices = [class_to_idx[cls] for cls in overlapping_classes if cls in class_to_idx]

    # Filter the training data for only the overlapping classes
    train_images = []
    train_labels = []
    for img, label in zip(data_train.data, data_train.targets):
        if label in overlapping_indices:
            train_images.append(img)
            train_labels.append(label)
    
    # Filter the test data for only the overlapping classes
    test_images = []
    test_labels = []
    for img, label in zip(data_test.data, data_test.targets):
        if label in overlapping_indices:
            test_images.append(img)
            test_labels.append(label)

    # Check the new class distribution
    train_class_counts = Counter(train_labels)
    test_class_counts = Counter(test_labels)
    print("Class distribution in the filtered training dataset:", train_class_counts)
    print("Class distribution in the filtered test dataset:", test_class_counts)

    # Return the Data object using the filtered datasets
    return Data(
        np.array(train_images),
        torch.LongTensor(train_labels),
        np.array(test_images),
        torch.LongTensor(test_labels),
        handler,
        args_task
    )
def get_TinyImageNet(handler, args_task):
    import cv2
    # URL to the Tiny ImageNet dataset (you might need to update this URL)
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    
    # Directory where the dataset will be extracted
    extract_to = "./data/TinyImageNet"
    os.makedirs(extract_to, exist_ok=True)
    download_and_unzip(url, extract_to)
    
    #download data from http://cs231n.stanford.edu/tiny-imagenet-200.zip and unzip it into ./data/TinyImageNet
    # deal with training set
    Y_train_t = []
    train_img_names = []
    train_imgs = []
    
    with open('./data/TinyImageNet/tiny-imagenet-200/wnids.txt') as wnid:
        for line in wnid:
            Y_train_t.append(line.strip('\n'))
    for Y in Y_train_t:
        Y_path = './data/TinyImageNet/tiny-imagenet-200/train/' + Y + '/' + Y + '_boxes.txt'
        train_img_name = []
        with open(Y_path) as Y_p:
            for line in Y_p:
                train_img_name.append(line.strip('\n').split('\t')[0])
        train_img_names.append(train_img_name)
    train_labels = np.arange(200)
    idx = 0
    for Y in Y_train_t:
        train_img = []
        for img_name in train_img_names[idx]:
            img_path = os.path.join('./data/TinyImageNet/tiny-imagenet-200/train/', Y, 'images', img_name)
            train_img.append(cv2.imread(img_path))
        train_imgs.append(train_img)
        idx = idx + 1
    train_imgs = np.array(train_imgs)
    train_imgs = train_imgs.reshape(-1, 64, 64, 3)
    X_tr = []
    Y_tr = []
    for i in range(train_imgs.shape[0]):
        Y_tr.append(i//500)
        X_tr.append(train_imgs[i])
    #X_tr = torch.from_numpy(np.array(X_tr))
    X_tr = np.array(X_tr)
    Y_tr = torch.from_numpy(np.array(Y_tr)).long()

    #deal with testing (val) set
    Y_test_t = []
    Y_test = []
    test_img_names = []
    test_imgs = []
    with open('./data/TinyImageNet/tiny-imagenet-200/val/val_annotations.txt') as val:
        for line in val:
            test_img_names.append(line.strip('\n').split('\t')[0])
            Y_test_t.append(line.strip('\n').split('\t')[1])
    for i in range(len(Y_test_t)):
        for i_t in range(len(Y_train_t)):
            if Y_test_t[i] == Y_train_t[i_t]:
                Y_test.append(i_t)
    test_labels = np.array(Y_test)
    test_imgs = []
    for img_name in test_img_names:
        img_path = os.path.join('./data/TinyImageNet/tiny-imagenet-200/val/images', img_name)
        test_imgs.append(cv2.imread(img_path))
    test_imgs = np.array(test_imgs)
    X_te = []
    Y_te = []

    for i in range(test_imgs.shape[0]):
        X_te.append(test_imgs[i])
        Y_te.append(Y_test[i])
    #X_te = torch.from_numpy(np.array(X_te))
    X_te = np.array(X_te)
    Y_te = torch.from_numpy(np.array(Y_te)).long()
    return Data(X_tr, Y_tr, X_te, Y_te, handler, args_task)
def get_ImageNet100(handler, args_task):
    import cv2
    # # URL to the Tiny ImageNet dataset (you might need to update this URL)
    # url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    
    # # Directory where the dataset will be extracted
    # extract_to = "./data/TinyImageNet"
    # os.makedirs(extract_to, exist_ok=True)
    # download_and_unzip(url, extract_to)
    
    #download data from http://cs231n.stanford.edu/tiny-imagenet-200.zip and unzip it into ./data/TinyImageNet
    # deal with training set
    Y_train_t = []
    train_img_names = []
    train_imgs = []
    
    with open('./data/TinyImageNet/tiny-imagenet-200/wnids.txt') as wnid:
        for line in wnid:
            Y_train_t.append(line.strip('\n'))
    for Y in Y_train_t:
        Y_path = './data/TinyImageNet/tiny-imagenet-200/train/' + Y + '/' + Y + '_boxes.txt'
        train_img_name = []
        with open(Y_path) as Y_p:
            for line in Y_p:
                train_img_name.append(line.strip('\n').split('\t')[0])
        train_img_names.append(train_img_name)
    train_labels = np.arange(200)
    idx = 0
    for Y in Y_train_t:
        train_img = []
        for img_name in train_img_names[idx]:
            img_path = os.path.join('./data/TinyImageNet/tiny-imagenet-200/train/', Y, 'images', img_name)
            train_img.append(cv2.imread(img_path))
        train_imgs.append(train_img)
        idx = idx + 1
    train_imgs = np.array(train_imgs)
    train_imgs = train_imgs.reshape(-1, 64, 64, 3)
    X_tr = []
    Y_tr = []
    for i in range(train_imgs.shape[0]):
        Y_tr.append(i//500)
        X_tr.append(train_imgs[i])
    #X_tr = torch.from_numpy(np.array(X_tr))
    X_tr = np.array(X_tr)
    Y_tr = torch.from_numpy(np.array(Y_tr)).long()

    #deal with testing (val) set
    Y_test_t = []
    Y_test = []
    test_img_names = []
    test_imgs = []
    with open('./data/TinyImageNet/tiny-imagenet-200/val/val_annotations.txt') as val:
        for line in val:
            test_img_names.append(line.strip('\n').split('\t')[0])
            Y_test_t.append(line.strip('\n').split('\t')[1])
    for i in range(len(Y_test_t)):
        for i_t in range(len(Y_train_t)):
            if Y_test_t[i] == Y_train_t[i_t]:
                Y_test.append(i_t)
    test_labels = np.array(Y_test)
    test_imgs = []
    for img_name in test_img_names:
        img_path = os.path.join('./data/TinyImageNet/tiny-imagenet-200/val/images', img_name)
        test_imgs.append(cv2.imread(img_path))
    test_imgs = np.array(test_imgs)
    X_te = []
    Y_te = []

    for i in range(test_imgs.shape[0]):
        X_te.append(test_imgs[i])
        Y_te.append(Y_test[i])
    #X_te = torch.from_numpy(np.array(X_te))
    X_te = np.array(X_te)
    Y_te = torch.from_numpy(np.array(Y_te)).long()
    return Data(X_tr, Y_tr, X_te, Y_te, handler, args_task)
def get_openml(handler, args_task, selection = 6):
    import openml
    from sklearn.preprocessing import LabelEncoder
    openml.config.apikey = '3411e20aff621cc890bf403f104ac4bc'
    openml.config.set_cache_directory('./data/openml/')
    ds = openml.datasets.get_dataset(selection)
    data = ds.get_data(target=ds.default_target_attribute)
    X = np.asarray(data[0])
    y = np.asarray(data[1])
    y = LabelEncoder().fit(y).transform(y)

    num_classes = int(max(y) + 1)
    nSamps, _ = np.shape(X)
    testSplit = .1
    inds = np.random.permutation(nSamps)
    X = X[inds]
    y = y[inds]

    split =int((1. - testSplit) * nSamps)
    while True:
        inds = np.random.permutation(split)
        if len(inds) > 50000: inds = inds[:50000]
        X_tr = X[:split]
        X_tr = X_tr[inds]
        X_tr = torch.Tensor(X_tr)

        y_tr = y[:split]
        y_tr = y_tr[inds]
        Y_tr = torch.Tensor(y_tr).long()

        X_te = torch.Tensor(X[split:])
        Y_te = torch.Tensor(y[split:]).long()

        if len(np.unique(Y_tr)) == num_classes: break
    return Data(X_tr, Y_tr, X_te, Y_te, handler, args_task)

def get_BreakHis(handler, args_task):
    # download data from https://www.kaggle.com/datasets/ambarish/breakhis and unzip it in data/BreakHis/
    data_dir = './data/BreaKHis_v1/BreaKHis_v1/histology_slides/breast'
    data = datasets.ImageFolder(root = data_dir, transform = None).imgs
    train_ratio = 0.7
    test_ratio = 0.3
    data_idx = list(range(len(data)))
    random.shuffle(data_idx)
    train_idx = data_idx[:int(len(data)*train_ratio)]
    test_idx = data_idx[int(len(data)*train_ratio):]
    X_tr = [np.array(Image.open(data[i][0])) for i in train_idx]
    Y_tr = [data[i][1] for i in train_idx]
    X_te = [np.array(Image.open(data[i][0])) for i in test_idx]
    Y_te = [data[i][1] for i in test_idx]
    X_tr = np.array(X_tr, dtype=object)
    X_te = np.array(X_te, dtype=object)
    Y_tr = torch.from_numpy(np.array(Y_tr))
    Y_te = torch.from_numpy(np.array(Y_te))
    return Data(X_tr, Y_tr, X_te, Y_te, handler, args_task)

def get_PneumoniaMNIST(handler, args_task):
    # download data from https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia and unzip it in data/PhwumniaMNIST/
    import cv2

    data_train_dir = './data/PneumoniaMNIST/chest_xray/train/'
    data_test_dir = './data/PneumoniaMNIST/chest_xray/test/'
    assert os.path.exists(data_train_dir)
    assert os.path.exists(data_test_dir)

    #train data
    train_imgs_path_0 = [data_train_dir+'NORMAL/'+f for f in os.listdir(data_train_dir+'/NORMAL/')]
    train_imgs_path_1 = [data_train_dir+'PNEUMONIA/'+f for f in os.listdir(data_train_dir+'/PNEUMONIA/')]
    train_imgs_0 = []
    train_imgs_1 = []
    for p in train_imgs_path_0:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (224, 224))
        train_imgs_0.append(im)
    for p in train_imgs_path_1:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (224, 224))
        train_imgs_1.append(im)
    train_labels_0 = np.zeros(len(train_imgs_0))
    train_labels_1 = np.ones(len(train_imgs_1))
    X_tr = []
    Y_tr = []
    train_imgs = train_imgs_0 + train_imgs_1
    train_labels = np.concatenate((train_labels_0, train_labels_1))
    idx_train = list(range(len(train_imgs)))
    random.seed(4666)
    random.shuffle(idx_train)
    X_tr = [train_imgs[i] for i in idx_train]
    Y_tr = [train_labels[i] for i in idx_train]
    X_tr = np.array(X_tr)
    Y_tr = torch.from_numpy(np.array(Y_tr)).long()

    #test data
    test_imgs_path_0 = [data_test_dir+'NORMAL/'+f for f in os.listdir(data_test_dir+'/NORMAL/')]
    test_imgs_path_1 = [data_test_dir+'PNEUMONIA/'+f for f in os.listdir(data_test_dir+'/PNEUMONIA/')]
    test_imgs_0 = []
    test_imgs_1 = []
    for p in test_imgs_path_0:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (224, 224))
        test_imgs_0.append(im)
    for p in test_imgs_path_1:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (224, 224))
        test_imgs_1.append(im)
    test_labels_0 = np.zeros(len(test_imgs_0))
    test_labels_1 = np.ones(len(test_imgs_1))
    X_te = []
    Y_te = []
    test_imgs = test_imgs_0 + test_imgs_1
    test_labels = np.concatenate((test_labels_0, test_labels_1))
    idx_test = list(range(len(test_imgs)))
    random.seed(4666)
    random.shuffle(idx_test)
    X_te = [test_imgs[i] for i in idx_test]
    Y_te = [test_labels[i] for i in idx_test]
    X_te = np.array(X_te)
    Y_te = torch.from_numpy(np.array(Y_te)).long()

    return Data(X_tr, Y_tr, X_te, Y_te, handler, args_task)


def get_waterbirds(handler, args_task):
    import wilds
    from torchvision import transforms
    dataset = wilds.get_dataset(dataset='waterbirds', root_dir='./data/waterbirds', download='True')
    trans = transforms.Compose([transforms.Resize([255,255])])
    train = dataset.get_subset(split = 'train',transform = trans)
    test = dataset.get_subset(split = 'test', transform = trans)

    len_train = train.metadata_array.shape[0]
    len_test = test.metadata_array.shape[0]
    X_tr = []
    Y_tr = []
    X_te = []
    Y_te = []

    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    f = open('waterbirds.txt', 'w')

    for i in range(len_train):
        x,y,meta = train.__getitem__(i)
        img = np.array(x)
        X_tr.append(img)
        Y_tr.append(y)

    for i in range(len_test):
        x,y, meta = test.__getitem__(i)
        img = np.array(x)

        X_te.append(img)
        Y_te.append(y)
        if meta[0] == 0 and meta[1] == 0:
            f.writelines('1') #landbird_background:land
            f.writelines('\n')
            count1 = count1 + 1
        elif meta[0] == 1 and meta[1] == 0:
            f.writelines('2') #landbird_background:water
            count2 = count2 + 1
            f.writelines('\n')
        elif meta[0] == 0 and meta[1] == 1:
            f.writelines('3') #waterbird_background:land
            f.writelines('\n')
            count3 = count3 + 1
        elif meta[0] == 1 and meta[1] == 1:
            f.writelines('4') #waterbird_background:water
            f.writelines('\n')
            count4 = count4 + 1
        else:
            raise NotImplementedError    
    f.close()

    Y_tr = torch.tensor(Y_tr)
    Y_te = torch.tensor(Y_te)
    X_tr = np.array(X_tr)
    X_te = np.array(X_te)
    return Data(X_tr, Y_tr, X_te, Y_te, handler, args_task)














