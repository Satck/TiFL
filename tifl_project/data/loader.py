import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10

def load_dataset(name):
    """加载数据集"""
    if name == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train[..., np.newaxis] / 255.0
        x_test = x_test[..., np.newaxis] / 255.0
    elif name == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train = x_train[..., np.newaxis] / 255.0
        x_test = x_test[..., np.newaxis] / 255.0
    elif name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train / 255.0
        x_test = x_test / 255.0
        y_train = y_train.squeeze()
        y_test = y_test.squeeze()
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    return (x_train, y_train), (x_test, y_test)

def create_non_iid_split(x, y, num_clients, shards_per_client=2):
    """
    Non-IID划分 (MNIST/Fashion-MNIST)
    每个客户端最多2个类别
    """
    # 按标签排序
    sorted_idx = np.argsort(y)
    x_sorted = x[sorted_idx]
    y_sorted = y[sorted_idx]
    
    # 分成100个shard
    num_shards = 100
    shard_size = len(x_sorted) // num_shards
    
    # 随机分配shard给客户端
    shard_indices = np.arange(num_shards)
    np.random.shuffle(shard_indices)
    
    client_data = []
    for i in range(num_clients):
        client_shards = shard_indices[i*shards_per_client:(i+1)*shards_per_client]
        client_x = []
        client_y = []
        for shard_id in client_shards:
            start = shard_id * shard_size
            end = start + shard_size if shard_id < num_shards - 1 else len(x_sorted)
            client_x.append(x_sorted[start:end])
            client_y.append(y_sorted[start:end])
        
        client_x = np.concatenate(client_x)
        client_y = np.concatenate(client_y)
        client_data.append((client_x, client_y))
    
    return client_data

def create_non_iid_cifar(x, y, num_clients, classes_per_client=5):
    """
    Non-IID划分 (CIFAR-10)
    每个客户端5个类别
    """
    num_classes = 10
    class_indices = [np.where(y == i)[0] for i in range(num_classes)]
    
    client_data = []
    for i in range(num_clients):
        # 为每个客户端随机选择5个类别
        selected_classes = np.random.choice(num_classes, classes_per_client, replace=False)
        client_idx = []
        for c in selected_classes:
            # 从每个类别中均匀取样
            samples_per_class = len(class_indices[c]) // (num_clients // 2)
            client_idx.extend(np.random.choice(class_indices[c], samples_per_class, replace=False))
        
        client_idx = np.array(client_idx)
        client_data.append((x[client_idx], y[client_idx]))
    
    return client_data
