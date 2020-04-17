import cv2
import copy
import numpy as np
import skimage.segmentation
import sklearn.metrics
from sklearn.linear_model import LinearRegression
import tensorflow as tf


class GradCAM(object):
    def __init__(self, model, layer=None, **kwargs):
        """
        Works only for Functional API, submodels are not allowed, Low level API would be implemented in the future
        :param model:
        :param layer:
        :param kwargs:
        """
        self.model = model
        self.reference_layer = layer
        self.grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer).output, model.output])

    def generate_explanation(self, img, label_idx):
        with tf.GradientTape() as tape:
            tape.watch(img)
            features, preds = self.grad_model(img)
            loss = preds[:, label_idx]

        output = features[0]
        grads = tape.gradient(loss, features)[0]
        gate_f = tf.cast(output > 0, 'float32')
        gate_r = tf.cast(grads > 0, 'float32')
        guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads

        weights = tf.reduce_mean(guided_grads, axis=(0, 1))

        cam = np.ones(output.shape[0: 2], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * output[:, :, i]

        cam = cv2.resize(cam.numpy(), (224, 224))
        cam = np.maximum(cam, 0)
        heatmap = (cam - cam.min()) / (cam.max() - cam.min())

        cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
        output_image = cv2.addWeighted(cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR), 0.5, cam, 1, 0)
        return output_image


class LIME(object):
    def __init__(self, model, areas=20, perturbations=700, **kwargs):
        self.model = model
        self.perturbations = perturbations
        self.areas = areas
        self.kernel_width = 0.25

    def create_perturbations(self, img, i, segments, create_visualization=False):
        active_pixels = np.where(i == 1)[0]
        mask = np.zeros(segments.shape)
        for act in active_pixels:
            mask[segments == act] = 1
        perturbed_img = copy.deepcopy(img)
        if create_visualization:
            # perturbed_img /= 255
            mask = mask.astype(np.float32)
            mask = mask[..., None]
            mask = np.concatenate((mask, mask, mask), axis=2)
            mask *= 0.2
            green_mark = np.ones(perturbed_img.shape, dtype=np.float32) * (0, 1, 0)
            perturbed_img = green_mark * mask + perturbed_img * (1.0 - mask)
        else:
            perturbed_img = perturbed_img * mask[:, :, np.newaxis]
        return perturbed_img

    def fit_linear_model(self, img, label):
        self.model.trainable = False
        # img = tf.cast([img], tf.float32)
        prediction = self.model(img)
        super_pixels = skimage.segmentation.quickshift(img[0].numpy(), kernel_size=2, ratio=0.1, max_dist=1000)
        num_super_pixels = np.unique(super_pixels).shape[0]
        perturbations = np.random.binomial(1, 0.5, size=(self.perturbations, num_super_pixels))
        preds = []
        for pert in perturbations:
            pert = self.create_perturbations(img[0].numpy(), pert, super_pixels)
            predictions = self.model(tf.cast([pert], tf.float32))
            preds.append(predictions.numpy()[0])
        preds = np.array(preds)
        initial_image = np.ones(num_super_pixels)[np.newaxis, :]

        distances = sklearn.metrics.pairwise_distances(perturbations, initial_image, metric='cosine').ravel()
        weights = np.sqrt(np.exp(-(distances ** 2) / self.kernel_width ** 2))

        y = preds[:, int(tf.argmax(label))]  # remove one hot
        linear_model = LinearRegression()
        linear_model.fit(X=perturbations, y=y, sample_weight=weights)
        coef = linear_model.coef_
        top_super_pixels = np.argsort(coef)[-self.areas:]
        mask = np.zeros(num_super_pixels)
        mask[top_super_pixels] = True
        explainer = self.create_perturbations(img[0].numpy(), mask, super_pixels, create_visualization=True)
        return explainer, mask, super_pixels





