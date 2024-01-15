import tensorflow as tf
import numpy as np
import os
import random
from zipfile import ZipFile
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

split = {0:'test',
        1:'train',
        2:'valid'}
def getMap(dataset):
    classes = [f for f in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, f))]
    class_map = {c:i for i,c in enumerate(classes)}
    return class_map

def make_chunk(data, labels, savepath, chunk_idx):
    print("Stacking the data and labels...")
    data = np.stack(data)
    labels = np.array(labels)
    print(data.shape, labels.shape)
    print(f"Saving Data to {savepath}_data_{chunk_idx}.npy...")
    np.save(f"{savepath}_data_{chunk_idx}.npy", data)
    print(f"Saving Data to {savepath}_label_{chunk_idx}.npy...")
    np.save(f"{savepath}_label_{chunk_idx}.npy", labels)
    


def process(dataset, data_path='data/', postfix='jpg', chunk_num=1000):
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    if not os.path.exists(f"{data_path}/{dataset[:-4]}"):
        os.mkdir(f"{data_path}/{dataset[:-4]}")
    classes = getMap(f"{data_path}/{dataset[:-4]}")
    file_list = [img for img in ZipFile(dataset, 'r').namelist() if img[-3:] == postfix]
    random.shuffle(file_list)
    train_list = file_list[:int(len(file_list)*0.8)]
    test_list = file_list[int(len(file_list)*0.8):int(len(file_list)*0.9)]
    valid_list = file_list[int(len(file_list)*0.9):]

    for i, imgs in enumerate((test_list, train_list, valid_list)):
        data = []
        labels = []
        chunk_idx = 0
        for j, f in enumerate(imgs):
            f = os.path.join(data_path, f)
            # 读取图像
            img = image.load_img(f, target_size=(720, 576))
            # 将图像转换为NumPy数组
            img = image.img_to_array(img)
            # 数据预处理
            img = preprocess_input(img)
            data.append(img)
            labels.append(classes[f.split('/')[-2]])
            print(j+1, '/', len(imgs), img.shape, labels[-1])
            if len(data) >= chunk_num:
                make_chunk(data, labels, f"{data_path}/{split[i]}", chunk_idx)
                data = []
                labels = []
                chunk_idx += 1
        make_chunk(data, labels, f"{data_path}/{split[i]}", chunk_idx)


def data_generator(data_file, label_file):
    data = np.load(data_file, mmap_mode='r')
    labels = np.load(label_file, mmap_mode='r')
    for d, l in zip(data, labels):
        yield d, l

def make_dataset(prefix, batch_size=32, data_path='data/', file_count=1, shuffle_buffer=10000):
    datasets = []
    for i in range(file_count):
        data_file = os.path.join(data_path, f'{prefix}_data_{i}.npy')
        label_file = os.path.join(data_path, f'{prefix}_label_{i}.npy')
        # Create dataset from generator
        dataset = tf.data.Dataset.from_generator(
            data_generator, 
            args=[data_file, label_file], 
            output_types=(tf.float32, tf.int32), 
            output_shapes=(tf.TensorShape([3,720,576]), tf.TensorShape([])) # Update shapes as per your data
        )
        datasets.append(dataset)
    # Concatenate all datasets
    dataset = datasets[0]
    for ds in datasets[1:]:
        dataset = dataset.concatenate(ds)
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def load_data(batch_size, chunks=[1,7,1]):
    loaders = []
    for i, prefix in split.items():
        loaders.append(make_dataset(prefix, batch_size, file_count=chunks[i]))
        print(prefix, "done")
    return loaders

if __name__ == '__main__':
    process('kvasir-dataset-v2.zip')
    # test, train, valid = load_data(32)
    pass