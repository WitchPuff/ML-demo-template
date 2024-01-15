import tensorflow as tf
import os
import random
import json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from bidict import bidict





def get_map(dataset):
    classes = [f for f in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, f))]
    class_map = {c:i for i,c in enumerate(classes)}
    return bidict(class_map)

def get_image(dataset_dir):
    paths = []
    labels = []
    class_map = get_map(dataset_dir)
    for class_name in class_map.keys():
        class_dir = os.path.join(dataset_dir, class_name)
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith(('jpg', 'jpeg', 'png')):
                paths.append(os.path.join(class_dir, img_file))
                labels.append(class_map[class_name])
    return paths, labels

def preprocess_image(path, label, img_size):
    img = image.load_img(path, target_size=img_size[:2])
    img = image.img_to_array(img)
    img = preprocess_input(img)
    return img, label

def data_generator(paths, labels, img_size):
    for path, label in zip(paths, labels):
        yield preprocess_image(path, label, img_size)

def make_dataset(paths, labels, batch_size, img_size):
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        args=[paths, labels, img_size],
        output_types=(tf.float32, tf.int32),
        output_shapes=(tf.TensorShape(img_size), tf.TensorShape([]))
    )
    return dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

def get_loaders(img_size, dataset_dir=r"data\kvasir-dataset-v2", batch_size=32):
    paths, labels = get_image(dataset_dir)
    data = list(zip(paths, labels))
    random.shuffle(data)
    train_end = int(len(data) * 0.8)
    valid_end = int(len(data) * 0.9)
    paths, labels = list(zip(*data))
    train_paths, valid_paths, test_paths = paths[:train_end], paths[train_end:valid_end], paths[valid_end:]
    train_labels, valid_labels, test_labels = labels[:train_end], labels[train_end:valid_end], labels[valid_end:]
    split = {
        'train': train_paths,
        'test': test_paths,
        'valid': valid_paths
    }
    with open('split.json', 'w') as f:
        json.dump(split, f)

    trainset = make_dataset(train_paths, train_labels, batch_size , img_size)
    validset = make_dataset(valid_paths, valid_labels, batch_size, img_size)
    testset = make_dataset(test_paths, test_labels, batch_size, img_size)

    return trainset, validset, testset

import matplotlib.pyplot as plt

# 测试数据处理代码
def test_data_processing():
    # 获取数据集加载器
    trainset, validset, testset = get_loaders()

    # 从训练集中获取第一批次数据
    for batch_data, batch_labels in trainset.take(1):
        # 打印第一批次的数据形状
        print("Batch Data Shape:", batch_data.shape)
        print("Batch Labels Shape:", batch_labels.shape)

        # 可视化第一个样本图像
        plt.figure(figsize=(4, 4))
        plt.imshow(batch_data[0].numpy())
        plt.title(f"Label: {batch_labels[0].numpy()}")
        plt.axis('off')
        plt.show()

