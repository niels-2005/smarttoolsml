def split_data(data, num_splits: int, split_size: int):
    """
    Splits a list of data into a specified number of parts, each of a given size.

    Args:
        data: The list of data to be split.
        num_splits (int): The number of parts to divide the data into.
        split_size (int): The number of elements each part should contain.

    Returns:
        splits: array, where each inner list represents a split of the 
                original data containing `split_size` elements, except possibly 
                for the last split.

    Example usage:
        data = [1, 2, 3, ..., 50000]
        splits = split_data(data, num_splits=5, split_size=10000)
        # This will split `data` into 5 parts, each with 10000 elements.
    """
    
    splits = [data[i*split_size:(i+1)*split_size] for i in range(num_splits)]

    for i, split in enumerate(splits, start=1):
        print(f'Länge von Split {i}: {len(split)}')

    return splits
