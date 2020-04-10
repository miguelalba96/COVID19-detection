import os
import argparse
from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

import utils
from net_tools import DataLoader


parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--model-name', help="Name of the CNN model trained")
parser.add_argument('--data-path', help="Folder containing the tf records")
parser.add_argument('--model-path', default='', help='Path of the model if was trained in other place')
parser.add_argument('--explain', action='store_true', default=False, help='Explanability methods ')

args = parser.parse_args()


class EvalDataset(object):
    def __init__(self, model_name, data_path, model_path=None, explain=False, **kwargs):
        if model_path is not None:
            self.model_path = os.path.join(model_path, model_name)
        else:
            self.model_path = os.path.join('/trained_models', model_name)

        self.model_name = model_name
        self.data = DataLoader(data_path, training=False).test_dataset()
        self.model = tf.keras.models.load_model(self.model_path)
        self.explain = explain
        self.outdir = os.path.join(model_path, 'results')
        self.class_names = ['normal', 'pneumonia', 'COVID-19']
        utils.mdir(self.outdir)

    def compute_metrics(self, predictions):
        results = dict(model_name=self.model_name, report=dict())
        for t in [0.5, 0.6, 0.7, 0.85, 0.9, 0.95]:
            filtered = []
            for ex in predictions:
                probabilities = ex['probabilities']
                predicted_as = ex['pred_class']
                if probabilities[predicted_as] >= t:
                    filtered.append(ex)

            conf_matrix = confusion_matrix(filtered['label'], filtered['pred_class'])

            meta = {
                'threshold_{}'.format(t): {
                    'confusion_matrix': conf_matrix,
                    'class_report': classification_report(filtered['label'],
                                                          filtered['pred_class'], target_names=self.class_names,
                                                          output_dict=True)
                }
            }
            results['report'].update(meta)
        return results

    def evaluate(self):
        predictions = []
        for batch in tqdm(self.data):
            imgs, labels = batch
            preds = self.model(batch, training=False)
            for i, ex in enumerate(imgs):
                meta = {
                    'img': imgs[i][:, :, 0].numpy(),
                    'probabilities': preds.numpy(),
                    'pred_class': int(tf.argmax(preds, axis=1)),
                    'label': int(tf.argmax(preds, axis=1))
                }
                if self.explain:
                    self.explanations()

                predictions.append(meta)
        results = self.compute_metrics(predictions)
        # TODO: add a saving in tf record for filename, to be used to check the direct images
        utils.save_json(results)
        utils.save(predictions, 'imgs.pdata')
        return predictions

    def explanations(self):
        raise NotImplementedError


if __name__ == '__main__':
    evaluation = EvalDataset(args.model_name, args.data_path, args.model_path, args.explain)
    evaluation.evaluate()
