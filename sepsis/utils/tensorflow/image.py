def plot_to_image(figure):
    """"Helper function to convert plot to image.

    Used in tensorflow callbacks to be able to store images.

    Usage
    -----
        figure = create_plot_matplotlib()
        image = plot_to_image(figure)
        tf.summary.image("name", cm_image, step=epoch)

    """
    # Libraries
    import io
    import tensorflow as tf
    import matplotlib.pyplot as plt

    # Convert
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    digit = tf.image.decode_png(buf.getvalue(), channels=4)
    digit = tf.expand_dims(digit, 0)

    # Return
    return digit