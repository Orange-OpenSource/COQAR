import json
import os

def json_get(file_path, default = None):
    if not os.path.exists(file_path):
        return default
    with open(file_path, 'r') as f:
        return json.load(f)

def json_save(thing, file_path, indent = -1):
    with open(file_path, 'w') as f:
        if indent == -1:
            json.dump(thing,f)
        else :
            json.dump(thing, f, indent = indent)

def get_file_path(dir_path, name):
    i = 0
    file_path = os.path.join(dir_path, name)
    while os.path.exists(file_path + str(i)):
        i += 1
    file_path = file_path + str(i)

    return file_path

def create_and_get_dir(upper_dir_path, name):
    i = 0
    dir_path = os.path.join(upper_dir_path, name)
    while os.path.exists(dir_path + str(i)):
        i += 1
    dir_path = dir_path + str(i)
    os.mkdir(dir_path)

    return dir_path


