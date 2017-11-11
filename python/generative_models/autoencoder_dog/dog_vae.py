"""DOG VAE"""
# import libs.gan as gan
import libs.vae as vae
import tensorflow as tf
from libs.datasets import DOG

if __name__ == '__main__':
    F, _ = DOG('data/train_vae')
    vae.train_vae(
        files=F,
        input_shape=[299, 299, 3],
        learning_rate=0.001,
        batch_size=400,
        n_epochs=500,
        crop_shape=[64, 64, 3],
        crop_factor=0.8,
        convolutional=True,
        variational=True,
        n_filters=[64, 128, 256],
        n_hidden=256,
        n_code=128,
        denoising=True,
        dropout=False,
        keep_prob=0.8,
        filter_sizes=[5, 3, 3],
        activation=tf.nn.sigmoid,
        ckpt_name='models/dog_vae.ckpt')
    # M = gan.VAEGAN(files=F, img_shape=(299, 299, 3), tf_dropout=True, tf_keep_prob=0.6)
    # M.train()
