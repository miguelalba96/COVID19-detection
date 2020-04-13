import tensorflow as tf

import conv_nets.layers as layers


class MiniVGG(tf.keras.Model):
    def __init__(self, name=None, **kwargs):
        super(MiniVGG, self).__init__(name=name)
        self.conv1 = layers.Conv2D(16, kernel_size=3, name='conv1', batch_norm=True, **kwargs)
        self.conv2 = layers.Conv2D(16, kernel_size=3, name='conv2', batch_norm=True, **kwargs)
        self.maxpool1 = layers.Pooling(type_pool='max', name='pool1')
        self.conv3 = layers.Conv2D(32, kernel_size=3, name='conv3', batch_norm=True, **kwargs)
        self.conv4 = layers.Conv2D(32, kernel_size=3, name='conv4', batch_norm=True, **kwargs)
        self.maxpool2 = layers.Pooling(type_pool='max', name='pool2')
        self.conv5 = layers.Conv2D(48, kernel_size=3, name='conv5', batch_norm=True, **kwargs)
        self.conv6 = layers.Conv2D(48, kernel_size=3, name='conv6', batch_norm=True, **kwargs)
        self.maxpool3 = layers.Pooling(type_pool='max', name='poo3')
        self.conv7 = layers.Conv2D(64, kernel_size=3, name='conv7', batch_norm=True, **kwargs)
        self.conv8 = layers.Conv2D(64, kernel_size=3, name='conv8', batch_norm=True, **kwargs)
        self.maxpool4 = layers.Pooling(type_pool='max', name='poo4')
        self.gap = tf.keras.layers.GlobalAveragePooling2D(name='gap')
        self.fc = tf.keras.layers.Dense(128, name='fc2', activation='relu')
        self.dp = tf.keras.layers.Dropout(0.5, name='dropout_2')
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
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.maxpool4(x)
        x = self.gap(x)
        x = self.fc(x)
        x = self.dp(x, training=training)
        x = self.prob(x)
        return x


class AttentionNet(tf.keras.Model):
    def __init__(self, name=None, **kwargs):
        super(AttentionNet, self).__init__(name=name)
        self.conv = layers.Conv2D(32, kernel_size=5, name='init_conv', batch_norm=True, **kwargs)
        self.block1 = layers.IdentityBlock([32, 32, 32], kernel_size=3, name='block1', **kwargs)
        self.pool1 = layers.Pooling(pool_size=2, name='pool1')
        self.block2 = layers.build_ResNeXt_block(filters=48, strides=1, groups=3, repeat_num=2,
                                                 name='block2', attention=True, **kwargs)
        self.pool2 = layers.Pooling(pool_size=2, name='pool2')
        self.block3 = layers.build_ResNeXt_block(filters=64, strides=1, groups=4, repeat_num=2,
                                                 name='block3', attention=True, **kwargs)
        self.pool3 = layers.Pooling(pool_size=2, name='pool2')
        self.dp = tf.keras.layers.SpatialDropout2D(0.5, name='dropout_1')
        self.gap = tf.keras.layers.GlobalAveragePooling2D(name='gap')
        self.prob = tf.keras.layers.Dense(3, activation='softmax', **kwargs)

    def call(self, inputs, training=False):
        x = inputs
        x = self.conv(x)
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.pool3(x)
        x = self.dp(x, training=training)
        x = self.gap(x)
        x = self.prob(x)
        return x
