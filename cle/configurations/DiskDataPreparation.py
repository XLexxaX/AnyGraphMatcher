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
    try:
        shutil.rmtree(path_to_cache)
    except:
        pass


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)
