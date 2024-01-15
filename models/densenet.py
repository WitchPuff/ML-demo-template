import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2D, AveragePooling2D, Concatenate
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Activation
from tensorflow.keras import Model

class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, num_conv, growth_rate, name):
        super(DenseBlock, self).__init__(name=name)
        self.num_conv = num_conv
        self.growth_rate = growth_rate
        self.layers_list = []

        for _ in range(self.num_conv):
            self.layers_list.append(self._conv_block())

    def call(self, x):
        for layer in self.layers_list:
            output = layer(x)
            x = Concatenate()([x, output])
        return x

    def _conv_block(self):
        layers = tf.keras.Sequential()
        layers.add(BatchNormalization())
        layers.add(Activation('relu'))
        layers.add(Conv2D(4 * self.growth_rate, kernel_size=1, use_bias=False))
        layers.add(BatchNormalization())
        layers.add(Activation('relu'))
        layers.add(Conv2D(self.growth_rate, kernel_size=3, padding='same', use_bias=False))
        return layers

class TransitionLayer(tf.keras.layers.Layer):
    def __init__(self, num_filters, name):
        super(TransitionLayer, self).__init__(name=name)
        self.num_filters = num_filters
        self.batch_norm = BatchNormalization()
        self.conv = Conv2D(self.num_filters, kernel_size=1, use_bias=False)
        self.avg_pool = AveragePooling2D(pool_size=2, strides=2)

    def call(self, x):
        x = self.batch_norm(x)
        x = Activation('relu')(x)
        x = self.conv(x)
        return self.avg_pool(x)

class DenseNet(Model):
    def __init__(self, input_shape=(224, 224, 3), num_classes=8):
        super(DenseNet, self).__init__()
        self.model = "DenseNet"
        self.input_size = input_shape
        
        self.conv1 = Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False)
        self.maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')
        
        # Dense Blocks and Transition Layers
        self.dense_block1 = DenseBlock(6, 32, name='dense_block1')
        self.transition1 = TransitionLayer(128, name='transition1')
        self.dense_block2 = DenseBlock(12, 32, name='dense_block2')
        self.transition2 = TransitionLayer(256, name='transition2')
        self.dense_block3 = DenseBlock(24, 32, name='dense_block3')
        self.transition3 = TransitionLayer(512, name='transition3')
        self.dense_block4 = DenseBlock(16, 32, name='dense_block4')

        # Classifier
        self.global_avg_pool = GlobalAveragePooling2D()
        self.classifier = Dense(num_classes, activation='softmax')
        
        

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        
        x = self.dense_block1(x)
        x = self.transition1(x)
        x = self.dense_block2(x)
        x = self.transition2(x)
        x = self.dense_block3(x)
        x = self.transition3(x)
        x = self.dense_block4(x)
        
        x = self.global_avg_pool(x)
        return self.classifier(x)
    


