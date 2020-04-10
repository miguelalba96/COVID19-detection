import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


class BoostAdvAttack(object):
    def __init__(self, model, iterations=10, momentum=1.0, epsilon=0.1):
        self.model = model
        self.loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.classes = 3
        self.epsilon = epsilon
        self.iterations = iterations
        self.momentum = momentum
        self.batch_size = 1

    def momentum_noise(self, img, gt, label, iterations, epsilon):
        alpha = epsilon/iterations
        with tf.GradientTape() as tape:
            tape.watch(img)
            preds = self.model(img)
            label = tf.argmin(tf.nn.softmax(preds), axis=1) if label is None else label
            loss = self.loss(tf.one_hot(label, self.classes), preds)
        grad = tape.gradient(loss, img)
        noise = grad / tf.reshape(K.std(tf.reshape(grad, [self.batch_size, -1]), axis=1), [self.batch_size, 1, 1, 1])
        velocity = (self.momentum*gt) + noise
        img = img + alpha*tf.sign(velocity)
        gt = grad
        return img, gt

    def boost_attack(self, input, label):
        img = np.array([input[..., None]])
        img = tf.convert_to_tensor(img)
        adv = img
        gt = tf.zeros(img.shape)
        for i in range(self.iterations):
            adv, gt = self.momentum_noise(adv, gt, label, self.iterations, self.epsilon)
        return adv, gt

