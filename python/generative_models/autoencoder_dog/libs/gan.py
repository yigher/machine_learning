"""VAEGAN"""
import os
import sys
import time
import tensorflow as tf
import numpy as np
import libs.utils as utils
from tensorflow.contrib.layers import batch_norm
from libs.dataset_utils import create_input_pipeline
from libs import utils


class VAEGAN(object):
    def __init__(
            self,
            files,
            img_shape,
            crop_shape=[64, 64, 3],
            batch_size=64,
            n_epochs=100,
            crop_factor=0.8,
            z_n_examples=10,
            model_dir='models',
            img_dir='data/model_imgs',
            input_shape=[None, 64, 64, 3],
            ae_channels=[100, 100, 100],
            ae_filter_sizes=[3, 3, 3],
            enc_activation=tf.nn.elu,
            enc_n_hidden=100,
            var_n_code=32,
            dec_channels=[100, 100, 100],
            dec_filter_sizes=[3, 3, 3, 3],
            dis_channels=[100, 100, 100, 100],
            dis_filter_sizes=[3, 3, 3, 3],
            learning_rate=0.0001,
            test_iter_number=50,
            save_model_iter=100,
            equilibrium=0.693,
            margin=0.4,
            tf_summary_flag=False,
            tf_restore=False,
            tf_dropout=False,
            tf_keep_prob=0.8):
        """"""
        self.files = files
        self.img_shape = img_shape
        print("img_shape: ", self.img_shape)
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.crop_shape = crop_shape
        print("crop_shape: ", self.crop_shape)
        self.crop_factor = crop_factor
        self.model_dir = model_dir
        self.img_dir = img_dir
        self.z_n_examples = z_n_examples
        print("z_n_examples: ", self.z_n_examples)
        self.input_shape = input_shape
        print("input_shape: ", self.input_shape)
        self.ae_channels = ae_channels
        print("ae_channels: ", self.ae_channels)
        self.ae_filter_sizes = ae_filter_sizes
        print("ae_filter_sizes: ", self.ae_filter_sizes)
        self.enc_activation = enc_activation
        self.enc_n_hidden = enc_n_hidden
        print("enc_n_hidden: ", self.enc_n_hidden)
        self.var_n_code = var_n_code
        print("var_n_code: ", self.var_n_code)
        self.dec_channels = dec_channels
        print("dec_channels: ", self.dec_channels)
        self.dec_filter_sizes = dec_filter_sizes
        print("dec_filter_sizes: ", self.dec_filter_sizes)
        self.dis_channels = dis_channels
        print("dis_channels: ", self.dis_channels)
        self.dis_filter_sizes = dis_filter_sizes
        print("dis_filter_sizes: ", self.dis_filter_sizes)
        self.learning_rate = learning_rate
        self.test_iter_number = test_iter_number
        self.save_model_iter = save_model_iter
        self.equilibrium = equilibrium
        self.margin = margin
        self.tf_summary_flag = tf_summary_flag
        self.tf_restore = tf_restore
        self.tf_dropout = tf_dropout
        self.tf_keep_prob = tf_keep_prob
        sys.stdout.flush()

    def encoder(self, x, is_training, channels, filter_sizes, activation=tf.nn.tanh, reuse=None):
        # Set the input to a common variable name, h, for hidden layer
        h = x

        print('encoder/input:', h.get_shape().as_list())
        # Now we'll loop over the list of dimensions defining the number
        # of output filters in each layer, and collect each hidden layer
        hs = []
        for layer_i in range(len(channels)):

            with tf.variable_scope('layer{}'.format(layer_i+1), reuse=reuse):
                # Convolve using the utility convolution function
                # This requirs the number of output filter,
                # and the size of the kernel in `k_h` and `k_w`.
                # By default, this will use a stride of 2, meaning
                # each new layer will be downsampled by 2.
                h, W = utils.conv2d(h, channels[layer_i],
                                    k_h=filter_sizes[layer_i],
                                    k_w=filter_sizes[layer_i],
                                    d_h=2,
                                    d_w=2,
                                    reuse=reuse)

                h = batch_norm(h, is_training=is_training)

                # Now apply the activation function
                h = activation(h)
                if self.tf_dropout:
                    h = tf.nn.dropout(h, self.tf_keep_prob)  
                print('layer:', layer_i, ', shape:', h.get_shape().as_list())

                # Store each hidden layer
                hs.append(h)

        # Finally, return the encoding.
        return h, hs

    def variational_bayes(self, h, n_code):
        # Model mu and log(\sigma)
        z_mu = tf.nn.tanh(utils.linear(h, n_code, name='mu')[0])
        z_log_sigma = 0.5 * tf.nn.tanh(utils.linear(h, n_code, name='log_sigma')[0])

        # Sample from noise distribution p(eps) ~ N(0, 1)
        epsilon = tf.random_normal(tf.stack([tf.shape(h)[0], n_code]))

        # Sample from posterior
        z = z_mu + tf.multiply(epsilon, tf.exp(z_log_sigma))

        # Measure loss
        loss_z = -0.5 * tf.reduce_sum(
            1.0 + 2.0 * z_log_sigma - tf.square(z_mu) - tf.exp(2.0 * z_log_sigma),
            1)

        return z, z_mu, z_log_sigma, loss_z

    def decoder(self, z, is_training, dimensions, channels, filter_sizes,
                activation=tf.nn.elu, reuse=None):
        h = z
        for layer_i in range(len(dimensions)):
            with tf.variable_scope('layer{}'.format(layer_i+1), reuse=reuse):
                h, W = utils.deconv2d(
                    x=h,
                    n_output_h=dimensions[layer_i],
                    n_output_w=dimensions[layer_i],
                    n_output_ch=channels[layer_i],
                    k_h=filter_sizes[layer_i],
                    k_w=filter_sizes[layer_i],
                    reuse=reuse)

                h = batch_norm(h, is_training=is_training)
                h = activation(h)
                if self.tf_dropout:
                    h = tf.nn.dropout(h, self.tf_keep_prob)
                print('layer:', layer_i, ', shape:', h.get_shape().as_list())
        return h


    def discriminator(
            self,
            X,
            is_training,
            channels=[50, 50, 50, 50],
            filter_sizes=[4, 4, 4, 4],
            activation=tf.nn.elu,
            reuse=None):
        # We'll scope these variables to "discriminator_real"
        with tf.variable_scope('discriminator', reuse=reuse):
            H, Hs = self.encoder(
                X, is_training, channels, filter_sizes, activation, reuse)
            shape = H.get_shape().as_list()
            print("H: ", H.get_shape().as_list())
            H = tf.reshape(
                H, [-1, shape[1] * shape[2] * shape[3]])
            print("H Reshaped: ", H.get_shape().as_list())
            D, W = utils.linear(
                x=H, n_output=1, activation=tf.nn.sigmoid, name='fc', reuse=reuse)
            print("D shape: ", D.get_shape().as_list())
        return D, Hs

    def create(self):
        n_code = self.var_n_code
        n_pixels = self.input_shape[1]
        print("n_pixels: ", n_pixels)
        X = tf.placeholder(name='X', shape=self.input_shape, dtype=tf.float32)
        sum_x = tf.summary.image("X", X)
        is_training = tf.placeholder(tf.bool, name='istraining')

        # encoder
        with tf.variable_scope('encoder'):
            print("===========encoder===========")
            H, Hs = self.encoder(
                x=X,
                is_training=is_training,
                channels=self.ae_channels,
                filter_sizes=self.ae_filter_sizes,
                activation=self.enc_activation,
                reuse=None)
            Z = utils.linear(H, self.enc_n_hidden)[0]
        # encoder's variational layer
        with tf.variable_scope('encoder/variational'):
            Z, Z_mu, Z_log_sigma, loss_Z = self.variational_bayes(h=Z, n_code=n_code)
        #decoder
        dimensions = []
        for i in range(len(self.ae_channels)+1):
            if i >= 1:
                dim_denominator = dim_denominator*2
            else:
                dim_denominator = 1
            dimensions.append(n_pixels//dim_denominator)
        dimensions = dimensions[::-1]
        # dimensions.append(n_pixels)
        print("decoder dimensions: ", dimensions)
        self.dec_channels.append(self.input_shape[-1])
        print("dec_channels dimensions: ", self.dec_channels)
        # to get [30, 30, 30, 4]
        channels = self.dec_channels
        filter_sizes = self.dec_filter_sizes
        print("filter_sizes: ", filter_sizes)
        activation = tf.nn.elu
        latent_denominator = 1
        for i in range(len(dimensions)):
            latent_denominator *= 2 
        print("latent_denominator: ", latent_denominator)
        n_latent = n_code * (n_pixels // latent_denominator)**2


        with tf.variable_scope('generator'):
            print("===========decoder===========")
            Z_decode = utils.linear(
                Z, n_output=n_latent, name='fc', activation=activation)[0]
            print('Z_decode shape:', Z_decode.get_shape().as_list())
            Z_decode_tensor = tf.reshape(
                Z_decode, [
                    -1,
                    n_pixels//latent_denominator,
                    n_pixels//latent_denominator,
                    n_code],
                name='reshape')
            print('Z_decode_tensor shape:', Z_decode_tensor.get_shape().as_list())
            G = self.decoder(
                Z_decode_tensor, is_training, dimensions,
                channels, filter_sizes, activation)
            sum_G = tf.summary.image("G", G)
        print("===========discriminator X===========")
        D_real, Hs_real = self.discriminator(
            X,
            is_training,
            channels=self.dis_channels,
            filter_sizes=self.dis_filter_sizes)
        print("===========discriminator G===========")
        D_fake, Hs_fake = self.discriminator(
            G,
            is_training,
            channels=self.dis_channels,
            filter_sizes=self.dis_filter_sizes,
            reuse=True)

        with tf.variable_scope('loss'):
            # Loss functions
            loss_D_llike = 0
            for h_real, h_fake in zip(Hs_real, Hs_fake):
                loss_D_llike += tf.reduce_sum(tf.squared_difference(
                    utils.flatten(h_fake), utils.flatten(h_real)), 1)

            eps = 1e-12
            loss_real = tf.log(D_real + eps)
            loss_fake = tf.log(1 - D_fake + eps)
            loss_GAN = tf.reduce_sum(loss_real + loss_fake, 1)

            gamma = 0.75
            loss_enc = tf.reduce_mean(loss_Z + loss_D_llike)
            loss_dec = tf.reduce_mean(gamma * loss_D_llike - loss_GAN)
            loss_dis = -tf.reduce_mean(loss_GAN)

        # Summaries

        sum_loss_enc = tf.summary.scalar("loss_enc", loss_enc)
        sum_loss_dec = tf.summary.scalar("loss_dec", loss_dec)
        sum_loss_dis = tf.summary.scalar("loss_dis", loss_dis)
        sum_loss_D = tf.summary.scalar("loss_D_llike", loss_D_llike)
        sum_D_real = tf.summary.histogram("D_real", D_real)
        sum_D_fake = tf.summary.histogram("D_fake", D_fake)
        sum_loss_real = tf.summary.histogram("loss_real", loss_real)
        sum_loss_fake = tf.summary.histogram("loss_fake", loss_fake)
        sys.stdout.flush()

        return {
            'X': X,
            'G': G,
            'Z': Z,
            'is_training': is_training,
            'loss_real': loss_real,
            'loss_fake': loss_fake,
            'loss_enc': loss_enc,
            'loss_dec': loss_dec,
            'loss_dis': loss_dis,
            'sums': {
                'sum_x': sum_x,
                'sum_G': sum_G,
                'sum_loss_enc': sum_loss_enc,
                'sum_loss_dec': sum_loss_dec,
                'sum_loss_dis': sum_loss_dis,
                'sum_loss_D': sum_loss_D,
                'sum_D_real': sum_D_real,
                'sum_D_fake': sum_D_fake,
                'sum_loss_real': sum_loss_real,
                'sum_loss_fake': sum_loss_fake
            }
        }

    def train(self):
        """train"""
        learning_rate = self.learning_rate
        model_file = os.path.join(self.model_dir, "vaegan.ckpt")
        batch = create_input_pipeline(
            files=self.files,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            crop_shape=self.crop_shape,
            crop_factor=self.crop_factor,
            shape=self.img_shape)

        model_out = self.create()

        sums = model_out['sums']
        opt_enc = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(
                model_out['loss_enc'],
                var_list=[
                    var_i for var_i in tf.trainable_variables()
                    if var_i.name.startswith('encoder')])

        opt_gen = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(
                model_out['loss_dec'],
                var_list=[
                    var_i for var_i in tf.trainable_variables()
                    if var_i.name.startswith('generator')])

        opt_dis = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(
                model_out['loss_dis'],
                var_list=[
                    var_i for var_i in tf.trainable_variables()
                    if var_i.name.startswith('discriminator')])

        # latent manifold
        zs = np.random.uniform(
            -1.0, 1.0, [4, self.var_n_code]).astype(np.float32)
        zs = utils.make_latent_manifold(zs, self.z_n_examples)

        # create a session to use the graph
        sess = tf.Session()
        init_op = tf.global_variables_initializer()

        saver = tf.train.Saver()

        if self.tf_summary_flag:
            G_sum_op = tf.summary.merge([
                sums['sum_G'], sums['sum_loss_enc'], sums['sum_loss_dec'],
                sums['sum_D_fake'], sums['sum_loss_fake']])
            D_sum_op = tf.summary.merge([
                sums['sum_D_real'], sums['sum_D_fake'],
                sums['sum_x'], sums['sum_loss_real'], sums['sum_loss_fake']])
            writer = tf.summary.FileWriter("./logs", sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(init_op)

        # model path
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        if self.tf_restore:
            saver.restore(sess, model_file)
            print("GAN model restored.")
        # image path
        n_files = len(self.files)
        test_xs = sess.run(batch) / 255.0
        if not os.path.exists(self.img_dir):
            os.mkdir(self.img_dir)
        utils.montage(test_xs, os.path.join(self.img_dir, 'test_xs.png'))
        # batch and epoch iteration numbers
        t_i = 0
        batch_i = 0
        epoch_i = 0
        ckpt_name = model_file.split('.')[0]
        # equilibrium and margin
        equilibrium = self.equilibrium
        margin = self.margin
        timestamp_1 = time.time()
        timestamp_2 = time.time()
        while epoch_i < self.n_epochs:
            if batch_i % (n_files // self.batch_size) == 0:
                batch_i = 0
                epoch_i += 1
                timestamp_1 = timestamp_2
                timestamp_2 = time.time()
                print("Iteration time: ", timestamp_2-timestamp_1, "s")
                print('epoch:', epoch_i)

            batch_i += 1
            batch_xs = sess.run(batch) / 255.0
            real_cost, fake_cost, _ = sess.run([
                model_out['loss_real'], model_out['loss_fake'], opt_enc],
                    feed_dict={
                        model_out['X']: batch_xs,
                        model_out['is_training']: True})
            real_cost = -np.mean(real_cost)
            fake_cost = -np.mean(fake_cost)

            gen_update = True
            dis_update = True

            if real_cost > (equilibrium + margin) or \
            fake_cost > (equilibrium + margin):
                gen_update = False

            if real_cost < (equilibrium - margin) or \
            fake_cost < (equilibrium - margin):
                dis_update = False

            if not (gen_update or dis_update):
                gen_update = True
                dis_update = True

            if gen_update:
                if self.tf_summary_flag:
                    _, sum_g = sess.run([opt_gen, G_sum_op], feed_dict={
                        model_out['X']: batch_xs,
                        model_out['is_training']: True})
                    writer.add_summary(sum_g, batch_i)
                else:
                    sess.run(opt_gen, feed_dict={
                        model_out['X']: batch_xs,
                        model_out['is_training']: True})
            if dis_update:
                if self.tf_summary_flag:
                    _, sum_d = sess.run([opt_dis, D_sum_op], feed_dict={
                        model_out['X']: batch_xs,
                        model_out['is_training']: True})
                    writer.add_summary(sum_d, batch_i)
                else:
                    sess.run(opt_dis, feed_dict={
                        model_out['X']: batch_xs,
                        model_out['is_training']: True})
            if batch_i % self.test_iter_number == 0:
                print('real:', real_cost, '/ fake:', fake_cost)
                # Plot example reconstructions from latent layer
                recon = sess.run(
                    model_out['G'], feed_dict={
                        model_out['Z']: zs,
                        model_out['is_training']: False})

                recon = np.clip(recon, 0, 1)
                utils.montage(
                    recon.reshape([-1] + self.crop_shape),
                    os.path.join(self.img_dir, 'manifold_%08d.png' % t_i))

                # Plot example reconstructions
                recon = sess.run(
                    model_out['G'], feed_dict={
                        model_out['X']: test_xs,
                        model_out['is_training']: False})
                recon = np.clip(recon, 0, 1)
                
                utils.montage(
                    recon.reshape([-1] + self.crop_shape),
                    os.path.join(self.img_dir, 'reconstruction_%08d.png' % t_i))
                t_i += 1


            if batch_i % self.save_model_iter == 0:
                # Save the variables to disk.
                save_path = saver.save(
                    sess, model_file)
                print("Model saved in file: %s" % save_path)
            sys.stdout.flush()
