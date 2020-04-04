import tensorflow as tf

import conv_nets.layers as layers


class COVID_VGGMini(tf.keras.Model):
    def __init__(self, name,**kwargs):
        super(COVID_VGGMini, self).__init__(name=name, **kwargs)
        self.conv1 = layers.Conv2D(32, kernel_size=5, name='conv1', **kwargs)
        self.conv2 = layers.Conv2D(32, kernel_size=3, name='conv2', **kwargs)
        self.maxpool1 = layers.Pooling(type_pool='max', name='pool1')
        self.conv3 = layers.Conv2D(64, kernel_size=3, name='conv3', **kwargs)
        self.conv4 = layers.Conv2D(64, kernel_size=3, name='conv4', **kwargs)
        self.maxpool2 = layers.Pooling(type_pool='max', name='pool2')
        self.conv5 = layers.Conv2D(96, kernel_size=3, name='conv5', **kwargs)
        self.conv6 = layers.Conv2D(96, kernel_size=3, name='conv6', **kwargs)
        self.maxpool3 = layers.Pooling(type_pool='max', name='poo3')
        self.flatten = tf.keras.layers.Flatten(name='flatten')
        self.fc1 = tf.keras.layers.Dense(128, name='fc1', activation='relu')
        self.dp1 = tf.keras.layers.Dropout(0.5, name='dropout_1')
        self.fc2 = tf.keras.layers.Dense(256, name='fc2', activation='relu')
        self.dp2 = tf.keras.layers.Dropout(0.5, name='dropout_2')
        self.prob = tf.keras.layers.Dense(3, activation='softmax', **kwargs)

    def call(self, inputs, training=False):
        x = inputs
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        if training:
            x = self.dp1(x)
        x = self.fc2(x)
        if training:
            x = self.dp2(x)
        x = self.prob(x)
        return x


class AttentionNet(tf.keras.Model):
    def __init__(self, name=None):
        super(AttentionNet, self).__init__()
        self.conv = None

    def call(self, inputs):
        x = inputs
        raise NotImplementedError
