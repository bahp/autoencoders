# Libraries
import tensorflow as tf

from utils.tensorflow.image import plot_to_image



class ConfusionMatrixLogger(tf.keras.callbacks.Callback):
    """

    """
    def __init__(self,  X, y, log_dir):
        """Constructor"""
        self.X = X
        self.y = y
        self.log_dir = log_dir
        self.file_writer = tf.summary.create_file_writer(str(log_dir))

    def plot_confusion_matrix(self):
        """Plot confusion matrix"""
        # Libraries
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        # Make predictions
        y_pred = self.model.predict(self.X) > 0.5

        # Create figure
        figure, axes = plt.subplots(1, 2, figsize=(8, 3))
        ConfusionMatrixDisplay.from_predictions(
            self.y, y_pred, cmap='Blues', normalize=None, ax=axes[0])
        ConfusionMatrixDisplay.from_predictions(
            self.y, y_pred, cmap='Blues', normalize='all', ax=axes[1])

        # Adjust size
        plt.tight_layout()

        # Return
        return figure

    def on_epoch_end(self, epoch, logs={}):
        """"""
        import matplotlib.pyplot as plt
        # Create figure
        figure = self.plot_confusion_matrix()
        # Convert to image
        cm_image = plot_to_image(figure)
        # Close
        plt.close("all")
        # Write summary.
        with self.file_writer.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)