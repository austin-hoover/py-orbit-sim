import argparse


parser = argparse.ArgumentParser("name")

parser.add_argument("--a", type=str, default="hi")
parser.add_argument("--b", type=int, default=1)
parser.add_argument("--c", type=int, default=2)
parser.add_argument("--d", type=str, default=3)

args = parser.parse_args()
print(args)