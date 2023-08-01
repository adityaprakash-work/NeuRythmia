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


# ---BRAIN2IMAGE----------------------------------------------------------------
class B2IVAE(tf.keras.Model):
    def __init__(self, name, input_shape, latent_dim, num_classes, **kwargs):
        super(B2IVAE, self).__init__(name=name, **kwargs)
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        inputs = tf.keras.Input(shape=self.input_shape)
        x = tf.keras.layers.Conv1D(
            32, 3, activation="relu", strides=2, padding="causal"
        )(inputs)
        x = tf.keras.layers.Conv1D(
            64, 3, activation="relu", strides=2, padding="causal"
        )(x)
        x = tf.keras.layers.Conv1D(
            128, 3, activation="relu", strides=2, padding="causal"
        )(x)
        x = tf.keras.layers.LSTM(128, return_sequences=True)(x)
        x = tf.keras.layers.LSTM(128)(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        z = tf.keras.layers.Dense(self.latent_dim + self.latent_dim)(x)
        return tf.keras.Model(inputs=inputs, outputs=z)

    def build_decoder(self):
        inputs = tf.keras.Input(shape=(self.latent_dim,))
        x = tf.keras.layers.Dense(
            self.latent_dim * self.latent_dim * 1, activation="relu"
        )(inputs)
        x = tf.keras.layers.Reshape((self.latent_dim, self.latent_dim, 1))
        x = tf.keras.layers.Conv2DTranspose(
            128,
            3,
            activation="relu",
            strides=2,
            padding="valid",
        )(x)
        x = tf.keras.layers.Conv2DTranspose(
            64,
            3,
            activation="relu",
            strides=2,
            padding="valid",
        )(x)
        x = tf.keras.layers.Conv2DTranspose(
            32,
            3,
            activation="relu",
            strides=2,
            padding="valid",
        )(x)
        y = tf.keras.layers.Conv2DTranspose(
            1,
            3,
            strides=1,
            padding="valid",
        )(x)
        return tf.keras.Model(inputs=inputs, outputs=y)

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        z_mean, z_logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return z_mean, z_logvar

    def reparameterize(self, z_mean, z_logvar):
        eps = tf.random.normal(shape=z_mean.shape)
        return eps * tf.exp(z_logvar * 0.5) + z_mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def call(self, x):
        z_mean, z_logvar = self.encode(x)
        z = self.reparameterize(z_mean, z_logvar)
        y = self.decode(z)
        # Add KL divergence regularization loss.
        kl_loss = -0.5 * tf.reduce_mean(
            z_logvar - tf.square(z_mean) - tf.exp(z_logvar) + 1
        )
        self.add_loss(kl_loss)
        return y


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
