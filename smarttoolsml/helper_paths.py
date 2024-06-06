def get_image_paths(image_path: str, format: str):
    """_summary_

    Args:
        image_path (str): _description_
        format (str): _description_

    Returns:
        _type_: _description_

    Example usage:
        data_path = Path("data/")
        image_path = data_path / "pizza_steak_sushi"
        format = "*/*/*.jpg"
        img_paths = get_image_paths(image_path=image_path, format=format)

        If you want to get a random path from list:
            random_path = random.choice(img_paths)
            img_class = random_image_path.parent.stem
    """
    img_paths_list = list(image_path.glob(format))
    return img_paths_list
