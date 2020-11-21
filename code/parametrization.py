import itertools
import sys
import yaml
import os
import datetime
import logging
import threading
import concurrent.futures

# for loopos.mkdir(save_path + folder_name)
def start_run(cuda, folder):
    print(cuda, "  ", folder)
    # os.system("ls -l")
    logging.info("Thread %s: starting", n)
    time.sleep(3)
    logging.info("Thread %s: finishing", n)


# read in arguments
arguments = sys.argv[:]

# yaml to be changed
default_yaml_path = open(os.path.abspath(arguments[1]), "r")
default_yaml = yaml.load(default_yaml_path)

# params to be changed
params_path = open(os.path.abspath(arguments[2]), "r")
params = yaml.load(params_path)

# save_path
if len(arguments) <= 3:
    now = str(datetime.datetime.now())
    save_path = "results/" + now.replace(" ", "_")
    os.mkdir(save_path)

else:
    save_path = arguments[3]


# create grid structure for given parameter
params_keys = [param for param in params.keys()]
params_list = [params[key] for key in params_keys]
grid = itertools.product(*params_list)


# grid_len = len(list(grid))


# list of folders to be submitted to main funciton
folder_list = []


# create folders for different runs and specify the corresponding config

for idx, elem in enumerate(grid):
    folder_name = ""
    for index, key in enumerate(params_keys):
        folder_name += params_keys[index] + ":" + str(elem[index]) + "_"
        default_yaml["TRAIN"][key] = elem[index]
    current_result_folder = save_path + "/" + folder_name
    os.mkdir(current_result_folder)
    file = open(current_result_folder + "/config.yaml", "w")
    yaml.dump(default_yaml, file)
    folder_list.append(current_result_folder)
    num_grid_elem = idx + 1


cuda_list = [0, 1] * int(float(num_grid_elem / 2))
# print(cuda_list)

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    executor.map(start_run, cuda, folder_list)
