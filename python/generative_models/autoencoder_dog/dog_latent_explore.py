import tensorflow as tf
from tensorflow.python.framework import ops
from libs.datasets import DOG
from libs.vae import VAE
from libs.dataset_utils import create_input_pipeline


ops.reset_default_graph()

input_shape=[299, 299, 3]
learning_rate=0.001
batch_size=100
n_epochs=500
crop_shape=[64, 64, 3]
crop_factor=0.8
convolutional=True
variational=True
n_filters=[64, 128, 256]
n_hidden=256
n_code=128
denoising=True
dropout=False
keep_prob=0.8
filter_sizes=[5, 3, 3]
activation=tf.nn.sigmoid

ae = VAE(input_shape=[None] + crop_shape,
             convolutional=convolutional,
             variational=variational,
             n_filters=n_filters,
             n_hidden=n_hidden,
             n_code=n_code,
             dropout=dropout,
             denoising=denoising,
             filter_sizes=filter_sizes,
             activation=activation)


ckpt_name = 'models/dog_vae.ckpt'

# We create a session to use the graph
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print("Restoring existing model")
saver.restore(sess, ckpt_name)

files, _ = DOG('data/train')
print("creating_input_pipeline")
batch = create_input_pipeline(
    files=files,
    batch_size=batch_size,
    n_epochs=n_epochs,
    crop_shape=crop_shape,
    crop_factor=crop_factor,
    shape=input_shape,
    n_threads=4)
print("normalising the batch")
batch_xs = sess.run(batch) / 255.0

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(batch_xs[0]), axs[0].grid('off'), axs[0].axis('off')


recon = sess.run(
    ae['y'],
    feed_dict={
        ae['x']: batch_xs,
        ae['train']: False,
        ae['keep_prob']: 1.0,
        ae['corrupt_prob']: [0.1]})

axs[1].imshow(recon[0]), axs[1].grid('off'), axs[1].axis('off')
