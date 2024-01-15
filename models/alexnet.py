from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

class AlexNet(Model):
    def __init__(self, input_shape=(227, 227, 3), num_classes=8):
        super(AlexNet, self).__init__()
        self.model = "AlexNet"
        self.input_size = input_shape
        self.conv1 = Conv2D(96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=input_shape)
        self.bn1 = BatchNormalization()
        self.pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))
        self.conv2 = Conv2D(256, kernel_size=(5, 5), activation='relu', padding='same')
        self.bn2 = BatchNormalization()
        self.pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))
        self.conv3 = Conv2D(384, kernel_size=(3, 3), activation='relu', padding='same')
        self.conv4 = Conv2D(384, kernel_size=(3, 3), activation='relu', padding='same')
        self.conv5 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')
        self.pool3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))
        self.flatten = Flatten()
        self.fc1 = Dense(4096, activation='relu')
        self.drop1 = Dropout(0.5)
        self.fc2 = Dense(4096, activation='relu')
        self.drop2 = Dropout(0.5)
        self.fc3 = Dense(num_classes, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    alexnet = AlexNet()
    # 打印模型概要
    alexnet.model.summary()