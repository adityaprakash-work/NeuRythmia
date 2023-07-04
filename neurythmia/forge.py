# ---INFO-----------------------------------------------------------------------
# Author(s):       Aditya Prakash
# Last Modified:   2023-06-30

# --Needed functionalities

# ---DEPENDENCIES---------------------------------------------------------------
import tensorflow as tf

from tqdm import tqdm


# ---STANDARD MODELS------------------------------------------------------------
class ConvLSTM(tf.keras.Model):
    def __init__(self, input_shape, num_classes):
        pass


# ---TRAINERS-------------------------------------------------------------------
class NRTrainer:
    """
    Class to train a model on a dataset.

    Parameters
    ----------
    x_pri : tf.data.Dataset
        Primary dataset
    x_aux : tf.data.Dataset
        Auxiliary dataset
    model : tf.keras.Model
        Model to be trained
    loss_fn : tf.keras.losses.Loss
        Loss function
    optimizer : tf.keras.optimizers.Optimizer
        Optimizer
    epochs : int
        Number of epochs
    metrics : list[tf.keras.metrics.Metric]
        List of metrics
    callbacks : list[tf.keras.callbacks.Callback]
        List of callbacks

    Returns
    -------
    trainer : NRTrainer
        Instance of the NRTrainer class
    """

    def __init__(
        self,
        x_pri,
        x_aux,
        model,
        loss_fn,
        optimizer,
        epochs=1,
        metrics=None,
        callbacks=None,
    ):
        self.x_pri = x_pri
        self.x_aux = x_aux
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epochs = epochs
        self.metrics = metrics
        if callbacks is not None:
            self.callbacks = tf.keras.callbacks.CallbackList(callbacks)
        else:
            self.callbacks = []
        self.x_pri_len = None
        self.x_aux_len = None

    def train(self):
        for epoch in range(self.epochs):
            # Primary dataset
            print(f"Epoch {epoch + 1}/{self.epochs}")
            if self.x_pri_len is None:
                print("Primary dataset: Calculating x_pri_len...")
                x_pri_len = 0
                for xb, yb in self.x_pri:
                    self.train_batch(xb, yb)
                    x_pri_len += 1
                self.x_pri_len = x_pri_len
            else:
                print("Primary dataset: Training...")
                for xb, yb in tqdm(self.x_pri, total=self.x_pri_len):
                    self.train_batch(xb, yb)
            for metric in self.metrics:
                print(f"{metric.name}: {metric.result().numpy()}")
                metric.reset_states()

            # Auxiliary dataset
            if self.x_aux is not None:
                if self.x_aux_len is None:
                    print("Auxiliary dataset: Calculating x_aux_len...")
                    x_aux_len = 0
                    for xb, yb in self.x_aux:
                        self.train_batch(xb, yb, training=False)
                        x_aux_len += 1
                    self.x_aux_len = x_aux_len
                else:
                    print("Auxiliary dataset: Validating...")
                    for xb, yb in tqdm(self.x_aux, total=self.x_aux_len):
                        self.train_batch(xb, yb, training=False)
                for metric in self.metrics:
                    print(f"{metric.name}: {metric.result().numpy()}")
                    metric.reset_states()

            print()

    def train_batch(self, xb, yb):
        with tf.GradientTape() as tape:
            yb_pred = self.model(xb, training=True)
            loss = self.loss_fn(yb, yb_pred)
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        for metric in self.metrics:
            metric.update_state(yb, yb_pred)


# ---METRICS--------------------------------------------------------------------
class ClsAcc(tf.keras.metrics.Accuracy):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, 1)
        y_pred = tf.argmax(y_pred, 1)
        return super(ClsAcc, self).update_state(y_true, y_pred, sample_weight)
