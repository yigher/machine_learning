""""Multi-variate Gaussian Distribution Fully Connected Neural Network"""
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf
import tensorflow.contrib.layers as tfl
import time
import libs.utils as utils

class MGD_NeuralNet(object):
    def __init__(
            self,
            layers=[
                (2, tf.nn.relu),
                (50, tf.nn.relu),
                (50, tf.nn.relu),
                (50, tf.nn.relu),
                (50, tf.nn.relu),
                (50, tf.nn.relu),
                (50, tf.nn.relu)],
            mgd_n_features=3,
            mgd_n_gaussians=10,
            mgd_gaussian_activation_fn=tf.nn.relu):
        """init"""
        self.layers = layers
        self.mgd_n_features = mgd_n_features
        self.mgd_n_gaussians = mgd_n_gaussians
        self.mgd_gaussian_activation_fn = mgd_gaussian_activation_fn

    def build(self, x_input=2, y_output=3):
        """build"""
        X = tf.placeholder(
            name='X', shape=[None, x_input],
            dtype=tf.float32)
        Y = tf.placeholder(
            name='Y', shape=[None, y_output],
            dtype=tf.float32)

        current_input = X
        for layer_i in range(len(self.layers)):
            current_input = utils.linear(
                current_input, self.layers[layer_i][0],
                activation=self.layers[layer_i][1],
                name='layer{}'.format(str(layer_i)))[0]

        cost = self.cost_mgd(current_input, Y)

        return {
            'X': X, 'Y': Y, 'cost': cost['cost'], 'means': cost['means'],
            'sigmas': cost['sigmas'], 'weights': cost['weights']}

    def train(
            self,
            xs, ys, img_shape=[128, 128, 3],
            batch_size=100,
            n_epochs=500,
            lr=0.001,
            save_epoch=1,
            optimizer_type=tf.train.AdamOptimizer):
        """train"""
        current_time = time.strftime('%Y%m%d_%H%M%S', time.gmtime())
        model_img_dir = "data/model_imgs/MGD_NeuralNet/" + current_time
        if not os.path.exists(model_img_dir):
            os.makedirs(model_img_dir)
        model_out = self.build()
        cost = model_out['cost']
        optimizer = optimizer_type(learning_rate=lr).minimize(cost)
        X = model_out['X']
        Y = model_out['Y']
        means = model_out['means']
        sigmas = model_out['sigmas']
        weights = model_out['weights']

        # We create a session to use the graph
        sess = tf.Session()
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        # This will handle our threaded image pipeline
        coord = tf.train.Coordinator()

        # Ensure no more changes to graph
        tf.get_default_graph().finalize()

        # Start up the queues for handling the image pipeline
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        t_i = 0
        batch_i = 0
        epoch_i = 0

        try:
            test_img = xs[0:(img_shape[0]*img_shape[1])]
            while not coord.should_stop() and epoch_i < n_epochs:
                idxs = np.random.permutation(range(len(xs)))
                n_batches = len(idxs) // batch_size
                print("n_batches: ", n_batches)
                train_cost = 0
                for batch_i in range(n_batches):
                    idxs_i = idxs[batch_i * batch_size: (batch_i + 1) * batch_size]
                    this_cost, _ = sess.run(
                        [cost, optimizer],
                        feed_dict={
                            X: xs[idxs_i],
                            Y: ys[idxs_i]})
                    train_cost += this_cost
                    if batch_i % 5000 == 0:
                        print("batch number: ", batch_i)

                print('iteration {}/{}: cost {}'.format(
                    epoch_i + 1, n_epochs, train_cost / n_batches))
                if (epoch_i + 1) % save_epoch == 0:
                    img = None
                    y_mu, y_dev, y_pi = sess.run(
                        [means, sigmas, weights],
                        feed_dict={X: test_img})
                    print("y_mu: ", y_mu.shape)
                    print("y_dev: ", y_dev.shape)
                    print("y_pi: ", y_pi.shape)
                    if False:
                        ys_pred = np.sum(y_mu * y_pi, axis=2)
                        img = np.clip(ys_pred, 0, 1)
                    else:
                        ys_pred = np.array([
                            y_mu[obv, :, idx]
                            for obv, idx in enumerate(np.argmax(y_pi.sum(1), 1))])
                        img = np.clip(ys_pred.reshape(img_shape), 0, 1)
                    if img is not None:
                        plt.imsave(
                            fname=os.path.join(
                                model_img_dir,
                                'recon_%08d.png' % epoch_i)
                            , arr=img)
                batch_i = 0
                epoch_i += 1

        except tf.errors.OutOfRangeError:
            print('Done.')
        finally:
            coord.request_stop()
            coord.join(threads)
            sess.close()

    def cost_mgd(
            self, current_input, Y, n_features=3,
            n_gaussians=5, mgd_activation_fn=tf.nn.relu):
        means = tf.reshape(
            tfl.linear(
                inputs=current_input,
                num_outputs=n_features * n_gaussians,
                activation_fn=mgd_activation_fn,
                scope='means'), [-1, n_features, n_gaussians])
        sigmas = tf.maximum(
            tf.reshape(
                tfl.linear(
                    inputs=current_input,
                    num_outputs=n_features * n_gaussians,
                    activation_fn=mgd_activation_fn,
                    scope='sigmas'), [-1, n_features, n_gaussians]), 1e-10)
        weights = tf.reshape(
            tfl.linear(
                inputs=current_input,
                num_outputs=n_features * n_gaussians,
                activation_fn=tf.nn.softmax,
                scope='weights'), [-1, n_features, n_gaussians])
        Y_3d = tf.reshape(Y, [-1, n_features, 1])
        p = utils.gausspdf(Y_3d, means, sigmas)
        weighted = weights * p
        sump = tf.reduce_sum(weighted, 2)
        negloglike = -tf.log(tf.maximum(sump, 1e-10) + 1)
        cost = tf.reduce_mean(tf.reduce_mean(negloglike, 1))

        return {'cost': cost, 'means': means, 'sigmas': sigmas, 'weights': weights}
