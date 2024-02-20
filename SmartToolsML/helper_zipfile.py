import os
import zipfile


def unzip_data(filename: str, extract_to: str = ".") -> None:
    """
    Unzips a zip file into a specified directory.

    This function takes a path to a zip file (filename) and extracts its contents into the specified
    directory (extract_to). If no directory is specified, the contents are extracted into the current
    working directory.

    Args:
        filename (str): The file path to the zip file that needs to be unzipped.
        extract_to (str): The target directory where the zip file's contents will be extracted.
                          Defaults to the current working directory (".").

    Returns:
        None

    Example usage:
        unzip_data(filename='path.zip', extract_to = '.')
    """
    # Ensure the target directory exists, create it if not
    os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(filename, "r") as zip_ref:
        zip_ref.extractall(path=extract_to)
        print(f"Extracted {filename} to {extract_to}")
