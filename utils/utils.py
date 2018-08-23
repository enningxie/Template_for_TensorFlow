import argparse


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--config',
        default='../configs/mnist.json',
        help='The configuration file.'
    )
    args = argparser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print(args.config)