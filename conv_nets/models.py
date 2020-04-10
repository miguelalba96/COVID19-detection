import tensorflow as tf

import conv_nets.layers as layers


class DebugNET(tf.keras.Model):
    def __init__(self, name=None, **kwargs):
        super(DebugNET, self).__init__(name=name)
        self.conv1 = layers.Conv2D(16, kernel_size=3, name='conv1', **kwargs)
        self.conv2 = layers.IdentityBlock([16, 16, 16], kernel_size=3, name='conv2', **kwargs)
        self.maxpool1 = layers.Pooling(type_pool='max', name='pool1')
        self.conv3 = layers.Conv2D(32, kernel_size=3, name='conv3', **kwargs)
        self.conv4 = layers.Conv2D(32, kernel_size=3, name='conv4', **kwargs)
        self.maxpool2 = layers.Pooling(type_pool='max', name='pool2')
        self.conv5 = layers.Conv2D(48, kernel_size=3, name='conv5', **kwargs)
        self.conv6 = layers.Conv2D(48, kernel_size=3, name='conv6', **kwargs)
        self.maxpool3 = layers.Pooling(type_pool='max', name='poo3')
        self.gap = tf.keras.layers.GlobalAveragePooling2D(name='gap')
        self.fc2 = tf.keras.layers.Dense(128, name='fc2', activation='relu')
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
        x = self.gap(x)
        x = self.fc2(x)
        x = self.dp2(x, training=training)
        x = self.prob(x)
        return x


class AttentionNet(tf.keras.Model):
    def __init__(self, name=None):
        super(AttentionNet, self).__init__()
        self.conv = None

    def call(self, inputs):
        x = inputs
        raise NotImplementedError
