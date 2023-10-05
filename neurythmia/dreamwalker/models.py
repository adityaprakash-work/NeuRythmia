# ---INFO-----------------------------------------------------------------------
# Author(s):       Aditya Prakash
# Last Modified:   2023-10-05

# --Needed functionalities
# 1. Reimplement models based on standard base

# ---DEPENDENCIES---------------------------------------------------------------
import tensorflow as tf
from forge import AnalyticalGaussianKLD
from .utils import plot_reconstructions

from tqdm import tqdm


# ---VARIATIONAL MODELS---------------------------------------------------------
class EncoderEmbeddingTutorI2E:
    def __init__(self, name, img_encoder, eeg_encoder, decoder):
        self.name = name
        self.img_encoder = img_encoder
        self.eeg_encoder = eeg_encoder
        self.decoder = decoder

        # Losses
        self.rec_loss = tf.keras.losses.MeanSquaredError()
        self.kld_loss = AnalyticalGaussianKLD()
        self.emb_loss = tf.keras.losses.CosineSimilarity()

        # Optimizers
        self.img_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.eeg_optimizer = tf.keras.optimizers.Adam(1e-4)

        # Metrics
        self.rec_metric = tf.keras.metrics.Mean()
        self.kld_metric = tf.keras.metrics.Mean()
        self.emb_metric = tf.keras.metrics.Mean()

    def switch_trainable(self, ie, ee, d):
        self.img_encoder.trainable = ie
        self.eeg_encoder.trainable = ee
        self.decoder.trainable = d

    def encode_img(self, xi):
        zi_mean, zi_log_var, zi = self.img_encoder(xi)
        return zi_mean, zi_log_var, zi

    def encode_eeg(self, xe):
        ze_mean, ze_log_var, ze = self.eeg_encoder(xe)
        return ze_mean, ze_log_var, ze

    def decode(self, z):
        xd = self.decoder(z)
        return xd

    @tf.function
    def train_step_i2i(self, xi):
        with tf.GradientTape() as tape:
            zi_mean, zi_log_var, zi = self.encode_img(xi)
            xd = self.decode(zi)
            rec_loss = self.rec_loss(xi, xd)
            kld_loss = self.kld_loss(zi_mean, zi_log_var)
            loss = rec_loss + kld_loss
        grads = tape.gradient(
            loss, self.img_encoder.trainable_weights + self.decoder.trainable_weights
        )
        self.img_optimizer.apply_gradients(
            zip(
                grads,
                self.img_encoder.trainable_weights + self.decoder.trainable_weights,
            )
        )
        self.rec_metric.update_state(rec_loss)
        self.kld_metric.update_state(kld_loss)
        return xd

    @tf.function
    def train_step_e2i(self, xe, xi):
        with tf.GradientTape() as tape:
            ze_mean, ze_log_var, ze = self.encode_eeg(xe)
            xd = self.decode(ze)
            rec_loss = self.rec_loss(xi, xd)
            kld_loss = self.kld_loss(ze_mean, ze_log_var)
            loss = rec_loss + kld_loss
        grads = tape.gradient(
            loss, self.eeg_encoder.trainable_weights + self.decoder.trainable_weights
        )
        self.eeg_optimizer.apply_gradients(
            zip(
                grads,
                self.eeg_encoder.trainable_weights + self.decoder.trainable_weights,
            )
        )
        self.rec_metric.update_state(rec_loss)
        self.kld_metric.update_state(kld_loss)
        return xd

    @tf.function
    def train_step_tutor(self, xe, xi):
        with tf.GradientTape() as tape:
            ze_mean, ze_log_var, ze = self.encode_eeg(xe)
            zi_mean, zi_log_var, zi = self.encode_img(xi)
            xd = self.decode(ze)
            rec_loss = self.rec_loss(xi, xd)
            kld_loss = self.kld_loss(ze_mean, ze_log_var)
            emb_loss = self.emb_loss(ze, zi)
            loss = rec_loss + kld_loss + emb_loss
        grads = tape.gradient(loss, self.eeg_encoder.trainable_weights)
        self.eeg_optimizer.apply_gradients(
            zip(
                grads,
                self.eeg_encoder.trainable_weights + self.decoder.trainable_weights,
            )
        )
        self.rec_metric.update_state(rec_loss)
        self.kld_metric.update_state(kld_loss)
        self.emb_metric.update_state(emb_loss)
        return xd

    def train(self, X_pri, X_aux=None, mode="i2i", epochs=1):
        if mode == "i2i":
            self.switch_trainable(True, False, True)
            for epoch in range(epochs):
                print(f"Epoch {epoch + 1}/{epochs}")
                for xi in tqdm(X_pri, desc="Primary dataset"):
                    xd = self.train_step_i2i(xi)
                print(
                    f"Rec Loss: {self.rec_metric.result().numpy()}, KLD Loss: {self.kld_metric.result().numpy()}"
                )
                self.rec_metric.reset_states()
                self.kld_metric.reset_states()
                plot_reconstructions(self, xi, xd)

        elif mode == "e2i":
            self.switch_trainable(False, True, True)
            for epoch in range(epochs):
                print(f"Epoch {epoch + 1}/{epochs}")
                for xe, xi in tqdm(X_pri, desc="Primary dataset"):
                    self.train_step_e2i(xe, xi)
                print(
                    f"Rec Loss: {self.rec_metric.result().numpy()}, KLD Loss: {self.kld_metric.result().numpy()}"
                )
                self.rec_metric.reset_states()
                self.kld_metric.reset_states()
                plot_reconstructions(self, xi, xd)

        elif mode == "tutor":
            self.switch_trainable(False, True, False)
            for epoch in range(epochs):
                print(f"Epoch {epoch + 1}/{epochs}")
                for xe, xi in tqdm(X_pri, desc="Primary dataset"):
                    self.train_step_tutor(xe, xi)
                print(
                    f"Rec Loss: {self.rec_metric.result().numpy()}, KLD Loss: {self.kld_metric.result().numpy()}, Emb Loss: {self.emb_metric.result().numpy()}"
                )
                self.rec_metric.reset_states()
                self.kld_metric.reset_states()
                self.emb_metric.reset_states()
                plot_reconstructions(self, xi, xd)
