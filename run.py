from utils.data import load_data
from utils.model import train_model, load_model
from utils.plot import plot_pca
import numpy as np
import argparse


def parse_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected')

    parser = argparse.ArgumentParser()

    parser.add_argument('--gs', type=str2bool, default=False,
                        help='Find optimal parameters with 10-Fold GridSearchCV')
    parser.add_argument('--load', type=str2bool, default=True,
                        help='True: Load trained model False: Train model default: False')

    parser.print_help()

    return parser.parse_args()


if __name__ == '__main__':
    np.random.seed(1)

    args = parse_args()

    data = load_data()

    plot_pca(data)

    digit = [0., 0., 0., 12., 13., 5., 0., 0.,
             0., 0., 0., 11., 16., 9., 0., 0.,
             0., 0., 3., 15., 16., 6., 0., 0.,
             0., 7., 15., 16., 16., 2., 0., 0.,
             0., 0., 1., 16., 16., 3., 0., 0.,
             0., 0., 1., 16., 16., 6., 0., 0.,
             0., 0., 1., 16., 16., 6., 0., 0.,
             0., 0., 0., 11., 16., 10., 0., 0.
             ]
    if args.load:
        model = load_model()
        print(model.predict(np.array(digit).reshape(1, -1))[0])

    else:

        model = train_model(data,args)
        print(model.predict(np.array(digit).reshape(1, -1))[0])
