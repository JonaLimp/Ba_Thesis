import itertools
import sys
import yaml
import os
import datetime


arguments = sys.argv[:]

default_yaml_path = open(os.path.abspath(arguments[1]), "r")
default_yaml = yaml.load(default_yaml_path)

params_path = open(os.path.abspath(arguments[2]), "r")
params = yaml.load(params_path)


if len(arguments) <= 3:
    save_path = "results"
else:
    save_path = arguments[3]


# print(default_yaml)
# load yaml
params_keys = [param for param in params.keys()]

params_list = [params[key] for key in params_keys]
grid = itertools.product(*params_list)

print(default_yaml)
for elem in grid:

    folder_name = ""
    for index, key in enumerate(params_keys):
        folder_name += params_keys[index] + ":" + str(elem[index]) + "_"
        default_yaml["TRAIN"][key] = elem[index]
    current_result_folder = save_path + "/" + folder_name
    os.mkdir(current_result_folder)
    file = open(current_result_folder + "/config.yaml", "w")
    yaml.dump(default_yaml, file)
    os.system(f"python3 code/main.py -c {current_result_folder}/config.yaml")
print(default_yaml)


# for loopos.mkdir(save_path + folder_name)
