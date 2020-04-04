import tensorflow as tf


class Conv2D(tf.keras.layers.Layer):
    def __init__(self, num_filters, kernel_size=3,  name='conv', strides=1, padding='same', depth_wise=False,
                 activ='relu', batch_norm=False, **kwargs):
        super(Conv2D, self).__init__(name=name)
        self.depth_wise = depth_wise
        self.num_filters = num_filters
        self.activation = activ
        self.batch_norm = batch_norm
        if self.depth_wise:
            self.conv = tf.keras.layers.DepthwiseConv2D(kernel_size, strides, padding,
                                                        depth_multiplier=num_filters,
                                                        name='depth_wise_conv', **kwargs)
        else:
            self.conv = tf.keras.layers.Conv2D(num_filters, kernel_size=kernel_size, strides=strides,
                                               name='conv1', padding=padding, **kwargs)

        if batch_norm:
            self.bn = tf.keras.layers.BatchNormalization(name='bn')
        if activ:
            self.activ = tf.keras.layers.Activation(activ, name=activ)

    def call(self, x):
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        if self.activation:
            x = self.activ(x)
        return x


class Pooling(tf.keras.layers.Layer):
    def __init__(self, type_pool='max', pool_size=(2, 2), name='pool', strides=None, **kwargs):
        super(Pooling, self).__init__(name=name)
        self.type_pool = type_pool
        if type_pool == 'avg':
            self.pool = tf.keras.layers.AveragePooling2D(pool_size, strides=strides, name='avg_pool', **kwargs)
        else:
            self.pool = tf.keras.layers.MaxPooling2D(pool_size, strides=strides, name='max_pool', **kwargs)

    def call(self, inputs):
        return self.pool(inputs)


class IdentityBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, name='identity', **kwargs):
        super(IdentityBlock, self).__init__(name=name)
        self.filters = filters
        self.conv1 = tf.keras.layers.Conv2D(filters[0], kernel_size=1, strides=1,
                                            name='conv1', padding='valid', **kwargs)
        self.bn1 = tf.keras.layers.BatchNormalization(name='bn1')
        self.relu1 = tf.keras.layers.ReLU(name='relu1')
        self.conv2 = tf.keras.layers.Conv2D(filters[1], kernel_size=kernel_size, strides=1,
                                            name='conv2', padding='same', **kwargs)
        self.bn2 = tf.keras.layers.BatchNormalization(name='bn2')
        self.relu2 = tf.keras.layers.ReLU(name='relu2')

        self.conv3 = tf.keras.layers.Conv2D(filters[2], kernel_size=1, strides=1,
                                            name='conv3', padding='valid', **kwargs)
        self.bn3 = tf.keras.layers.BatchNormalization(name='bn3')
        self.relu3 = tf.keras.layers.ReLU(name='relu3')
        self.relu = tf.keras.layers.ReLU(name='relu')

    def call(self, inputs):
        x = inputs
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x += inputs
        return self.relu(x)


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, name='block', strides=2,**kwargs):
        super(ConvBlock, self).__init__(name=name)
        self.filters = filters
        self.conv1 = tf.keras.layers.Conv2D(self.filters[0], kernel_size=1, strides=strides,
                                            name='conv1', padding='valid', **kwargs)
        self.bn1 = tf.keras.layers.BatchNormalization(name='bn1')
        self.relu1 = tf.keras.layers.ReLU(name='relu1')
        self.conv2 = tf.keras.layers.Conv2D(self.filters[1], kernel_size=kernel_size, strides=1,
                                            name='conv2', padding='same', **kwargs)
        self.bn2 = tf.keras.layers.BatchNormalization(name='bn2')
        self.relu2 = tf.keras.layers.ReLU(name='relu2')

        self.conv3 = tf.keras.layers.Conv2D(self.filters[2], kernel_size=1, strides=1,
                                            name='conv3', padding='valid', **kwargs)
        self.bn3 = tf.keras.layers.BatchNormalization(name='bn3')
        self.relu3 = tf.keras.layers.ReLU(name='relu3')

        self.conv4 = tf.keras.layers.Conv2D(self.filters[2], kernel_size=1, strides=strides,
                                            name='conv4', padding='valid', **kwargs)
        self.bn4 = tf.keras.layers.BatchNormalization(name='bn4')
        self.relu = tf.keras.layers.ReLU(name='relu')

    def call(self, inputs):
        x = inputs
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        inputs = self.conv4(inputs)
        inputs = self.bn4(inputs)
        x += inputs
        return self.relu(x)



