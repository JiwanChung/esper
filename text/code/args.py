import argparse

def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transformer', default='gpt2', type=str)
    parser.add_argument('--max-length', default=20, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--use-kl', action='store_true')
    parser.add_argument('--add_eot', action='store_true')
    parser.add_argument("--filenames", nargs="+", default=None)
    parser.add_argument("--no_prefix_label", action='store_true')
    args = parser.parse_args()

    return args
