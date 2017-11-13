"""DOG VAE"""
# import libs.gan as gan
import tensorflow as tf
from libs.vae_class import VAE
from libs.datasets import DOG

if __name__ == '__main__':
    F, _ = DOG('data/train')
    V = VAE(files=F,
            input_shape=[299, 299, 3],
            learning_rate=0.001,
            batch_size=400,
            n_epochs=50,
            n_examples=10,
            crop_shape=[64, 64, 3],
            crop_factor=0.8,
            encoder_filters=[
                (64, 5, 2, tf.nn.sigmoid),
                (128, 3, 2, tf.nn.sigmoid),
                (256, 3, 2, tf.nn.sigmoid)],
            encoder_pool=None,
                # [
                # (tf.nn.max_pool, 5, 2),
                # (tf.nn.max_pool, 3, 2),
                # (tf.nn.max_pool, 3, 2)],
            n_fc_hidden=256,
            n_latent_samples=128,
            variational=True,
            decoder_filters=[
                (128, 3, tf.nn.sigmoid),
                (64, 3, tf.nn.sigmoid),
                (3, 3, tf.nn.sigmoid)],
            denoising=False,
            dropout=True,
            keep_prob=0.8,
            corrupt_prob=0.5,
            activation=tf.nn.sigmoid,
            img_step=100,
            save_step=100,
            ckpt_name="vae.ckpt")
    V.train()
