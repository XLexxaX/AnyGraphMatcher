import shutil
import os

def prepare_file(path):
    if not os.path.exists(path):
        return

    try:
        os.remove(path+".tmp")
    except FileNotFoundError:
        pass

    shutil.copyfile(path, path+".tmp")
    os.remove(path)
    f = open(path, "w+", encoding="UTF-8")
    with open(path+".tmp", encoding="UTF-8") as file:
        for line in file:
            f.write(line.lower())
    f.flush()
    f.close()
    os.remove(path + ".tmp")

def prepare_dir(path):
    if os.path.exists(path):
        return

    os.mkdir(path)

def clean_cache(path_to_cache):
    shutil.rmtree(path_to_cache)
