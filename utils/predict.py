import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from .data import get_map
import numpy as np
import os

def predict(model, image_path, dataset='data\kvasir-dataset-v2'):
    w, h, c = model.input_size
    dummy_input = tf.zeros((1, w, h, c))
    # 确保模型被构建
    if not model.built:
        model.build((None, w, h, c))
        model(dummy_input)  # 或者仅仅调用模型
    model_path=f'models\{model.model}_best.h5'
    if not os.path.exists(model_path):
        print("No pretrained weights for this model.")
        return 
    print("Loading weights from previously trained model.")
    model.load_weights(model_path)
        
    class_map = get_map(dataset)
    print(class_map)
    # 加载图像，调整大小至模型所需尺寸
    img = image.load_img(image_path, target_size=model.input_size)
    # 将图像转化为数组
    img_array = image.img_to_array(img)
    # 添加一个维度，将数组转化为网络输入所需形状的批次
    img_batch = np.expand_dims(img_array, axis=0)
    # 预处理图像
    img_preprocessed = preprocess_input(img_batch)
    # 进行预测
    prediction = class_map.inv[np.argmax(model.predict(img_preprocessed))]
    print('Image path: ', image_path)
    print('Predict: ', prediction)
    # 解码预测结果
    return prediction  # 返回最可能的标签

# 使用示例
