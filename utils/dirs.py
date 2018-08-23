import os


def create_dirs(dirs):
    # dirs - a list of directories to create if these directories are not found
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {}".format(err))
        exit(-1)


if __name__ == '__main__':
    dirs = ['../1', '../2']
    create_dirs(dirs)