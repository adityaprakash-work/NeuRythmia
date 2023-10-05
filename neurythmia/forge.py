# ---INFO-----------------------------------------------------------------------
# Author(s):       Aditya Prakash
# Last Modified:   2023-06-30

# --Needed functionalities

# ---DEPENDENCIES---------------------------------------------------------------
import tensorflow as tf

from tqdm import tqdm


# ---STANDARD MODELS------------------------------------------------------------
class VariationalImageEncoder(tf.keras.Model):
    def __init__(self, name, latent_dim=128):
        super(VariationalImageEncoder, self).__init__(name=name)
        self.latent_dim = latent_dim
        self.conv1 = tf.keras.layers.Conv2D(
            32, 3, strides=2, padding="same", activation="relu"
        )
        self.conv2 = tf.keras.layers.Conv2D(
            64, 3, strides=2, padding="same", activation="relu"
        )
        self.conv3 = tf.keras.layers.Conv2D(
            128, 3, strides=2, padding="same", activation="relu"
        )
        self.flatten = tf.keras.layers.Flatten()
        self.mu = tf.keras.layers.Dense(latent_dim)
        self.log_var = tf.keras.layers.Dense(latent_dim)
        self.z = tf.keras.layers.Lambda(self.reparameterize)

    def reparameterize(self, args):
        mean, log_var = args
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(log_var * 0.5) + mean

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        mean = self.mu(x)
        log_var = self.log_var(x)
        z = self.z([mean, log_var])
        return mean, log_var, z


class VariationalEEGEncoder(tf.keras.Model):
    def __init__(self, name, latent_dim=128):
        super(VariationalEEGEncoder, self).__init__(name=name)
        self.latent_dim = latent_dim
        self.lstm = tf.keras.layers.LSTM(128)
        self.mu = tf.keras.layers.Dense(latent_dim)
        self.log_var = tf.keras.layers.Dense(latent_dim)
        self.z = tf.keras.layers.Lambda(self.reparameterize)

    def reparameterize(self, args):
        mean, log_var = args
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(log_var * 0.5) + mean

    def call(self, x):
        x = self.lstm(x)
        mean = self.mu(x)
        log_var = self.log_var(x)
        z = self.reparameterize([mean, log_var])
        return mean, log_var, z


class ImageDecoder(tf.keras.Model):
    def __init__(self, name):
        super(ImageDecoder, self).__init__(name=name)
        self.dense = tf.keras.layers.Dense(4 * 4 * 128, activation="relu")
        self.reshape = tf.keras.layers.Reshape((4, 4, 128))
        self.convT1 = tf.keras.layers.Conv2DTranspose(
            128, 3, strides=2, padding="same", activation="relu"
        )
        self.convT2 = tf.keras.layers.Conv2DTranspose(
            64, 3, strides=2, padding="same", activation="relu"
        )
        self.convT3 = tf.keras.layers.Conv2DTranspose(
            32, 3, strides=2, padding="same", activation="relu"
        )
        self.convT4 = tf.keras.layers.Conv2DTranspose(
            3, 3, strides=1, padding="same", activation="sigmoid"
        )

    def call(self, z):
        x = self.dense(z)
        x = self.reshape(x)
        x = self.convT1(x)
        x = self.convT2(x)
        x = self.convT3(x)
        x = self.convT4(x)
        return x


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
            self.on_epoch_end()

    def on_epoch_end(self):
        self.x_aux = self.x_aux.shuffle(self.x_aux_len)
        self.x_pri = self.x_pri.shuffle(self.x_pri_len)

    def train_batch(self, xb, yb, training=True):
        if training:
            with tf.GradientTape() as tape:
                yb_pred = self.model(xb, training=training)
                loss = self.loss_fn(yb, yb_pred)
            grads = tape.gradient(loss, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        else:
            yb_pred = self.model(xb, training=training)
        for metric in self.metrics:
            metric.update_state(yb, yb_pred)


# ---METRICS--------------------------------------------------------------------
class ClsAcc(tf.keras.metrics.Accuracy):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, 1)
        y_pred = tf.argmax(y_pred, 1)
        return super(ClsAcc, self).update_state(y_true, y_pred, sample_weight)


# ---LOSSES---------------------------------------------------------------------
class AnalyticalGaussianKLD:
    """
    # Tensorflow 2.0's built-in KLDivergence loss defines it as:
    # `loss = y_true * log(y_true / y_pred)`
    # Following is the analytical formula for the KL divergence between two
    # Gaussian distributions:
    # `loss = 0.5 * (log(sigma2) - log(sigma1) + (sigma1^2 + (mu1 - mu2)^2) / sigma2^2 - 1)`

    Source - https://math.stackexchange.com/questions/2888353/how-to-analytically-compute-kl-divergence-of-two-gaussian-distributions
    """

    def __call__(self, mean1, log_var1, mean2=0.0, log_var2=0.0):
        term1 = log_var2 - log_var1
        term2 = (tf.exp(log_var1) + tf.square(mean1 - mean2)) / tf.exp(log_var2) - 1.0
        kl_divergence = 0.5 * (term1 + term2)
        return tf.reduce_mean(kl_divergence)
