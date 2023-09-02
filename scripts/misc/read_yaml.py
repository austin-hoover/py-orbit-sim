import os
import yaml


filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.yaml")
file = open(filename, "r")
data = yaml.safe_load(file)
file.close()

print(data)