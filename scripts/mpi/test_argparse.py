import argparse


parser = argparse.ArgumentParser('OT-Flow')

parser.add_argument("--arg", type=str, default="hi")

args = parser.parse_args()
print(args)