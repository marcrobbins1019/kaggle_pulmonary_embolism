import argparse

import pandas as pd

from models import models
from train_utils import PE3DGenerator

parser = argparse.ArgumentParser()

parser.add_argument('-dir', "--cache_dir")
parser.add_argument('-model', '--model')

parser.add_argument('-tq', "--train_csv")
parser.add_argument('-vq', "--validation_csv")

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

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.validation_csv)

    generator = PE3DGenerator(train_df)



    # TODO figure out how to use with generator?
    #TODO x and y different?
    model.fit(x=generator)  # TODO
    # TODO build train generator


if __name__ == "__main__":
    main()
