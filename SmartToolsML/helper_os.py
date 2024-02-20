import os


def walk_through_dir(dir_path: str) -> None:
    """
    Walks through dir_path returning its contents.

    Args:
      dir_path (str): target directory

    Returns:
      A print out of:
        number of subdiretories in dir_path
        number of images (files) in each subdirectory
        name of each subdirectory

    Example usage:
      walk_through_dir(dir_path='./directory')
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(
            f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'."
        )
