import tensorflow as tf


def get_tf_dataset(
    X_train, y_train, batch=True, batch_size=16, scale=True, shuffle=True
):
    ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(X_train))

    if batch:
        ds = ds.batch(batch_size=batch_size)

    if scale:
        # scale X to [-1, 1]
        ds = ds.map(lambda x, y: (x * 2 - 1.0, y))

    return ds
