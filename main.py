import argparse

from kaggle_pulmonary_embolism.models import models

parser = argparse.ArgumentParser()

parser.add_argument('-dir', "--cache_dir")
parser.add_argument('-model', '--model')

parser.add_argument('-window', '--window', default=(400, 40), type=int)
parser.add_argument("-n_dim", "--number_dimensions", default=3, choices=[2, 3])
parser.add_argument('-dim', '--dimension', default=(40, 256, 256), type=int,
                    help="Representation Dimension (z,y,x) or (x,y)")

parser.add_argument('-lr', '--learning_rate', default=1e-4, help="Optimizer Learning Rate")
parser.add_argument('-bs', '--batch_size', default=8)
parser.add_argument('-spe', '--steps_per_epoch', default=150)
parser.add_argument('-e', '--epochs', default=1000)
parser.add_argument('-aug', '--augmentation', default='standard')


def main(argv=None):
    args = parser.parse_args()
    model = models[args.model]
    model.compile(optimizer='adam',
                  metrics=['accuracy', 'roc_auc'])  # TODO
    # TODO figure out how to use with generator?
    model.fit()  # TODO
    # TODO build train generator


if __name__ == "__main__":
    main()
