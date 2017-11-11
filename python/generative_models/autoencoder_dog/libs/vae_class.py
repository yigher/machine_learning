"""VAE class"""
import tensorflow as tf
import numpy as np
import os
import sys
import time
from libs.dataset_utils import create_input_pipeline
from libs.batch_norm import batch_norm
from libs import utils


class VAE(object):
    """initialise VAE object"""
    def __init__(
            self,
            files,
            input_shape,
            tf_summary_flag=False,
            learning_rate=0.0001,
            batch_size=100,
            n_epochs=50,
            n_examples=10,
            crop_shape=[64, 64, 3],
            crop_factor=0.8,
            encoder_filters=[
                (32, 3, 1, tf.nn.sigmoid),
                (64, 3, 1, tf.nn.sigmoid),
                (128, 3, 1, tf.nn.sigmoid)],
            encoder_pool=[(tf.nn.max_pool, 3, 2), (tf.nn.max_pool, 3, 2), (tf.nn.max_pool, 3, 2)],
            n_fc_hidden=256,
            n_latent_samples=50,
            variational=True,
            decoder_filters=[
                (64, 3, tf.nn.sigmoid),
                (128, 3, tf.nn.sigmoid),
                (3, 3, tf.nn.sigmoid)],
            denoising=False,
            dropout=True,
            keep_prob=0.8,
            corrupt_prob=0.5,
            activation=tf.nn.relu,
            img_step=100,
            save_step=100,
            recon_number=100,
            ckpt_name="vae.ckpt"):
        """init"""
        """General purpose training of a (Variational) (Convolutional) Autoencoder.
        Supply a list of file paths to images, and this will do everything else.
        Parameters
        ----------
        files : list of strings
            List of paths to images.

        input_shape : list
            Must define what the input image's shape is.

        learning_rate : float, optional
            Learning rate.

        batch_size : int, optional
            Batch size.

        n_epochs : int, optional
            Number of epochs.

        n_examples : int, optional
            Number of example to use while demonstrating the current training
            iteration's reconstruction.  Creates a square montage, so make
            sure int(sqrt(n_examples))**2 = n_examples, e.g. 16, 25, 36, ... 100.

        crop_shape : list, optional
            Size to centrally crop the image to.

        crop_factor : float, optional
            Resize factor to apply before cropping.

        n_filters : list, optional
            Same as VAE's n_filters.

        encoder_pool: optional
            pool configuration stored
            in a list as a tuple (pooling_type, filter_size, stride)
            replace with None in the list if the particular layer does not require pooling
            i.e. [(3, 2), None, (3, 3)]

        variational : bool, optional
            Whether or not to create a variational embedding layer.  This will
            create a fully connected layer after the encoding, if `n_hidden` is
            greater than 0, then will create a multivariate gaussian sampling
            layer, then another fully connected layer.  The size of the fully
            connected layers are determined by `n_hidden`, and the size of the
            sampling layer is determined by `n_code`.

        n_fc_hidden: int
            first fully
            connected layer prior to the variational embedding, directly after
            the encoding.  After the variational embedding, another fully connected
            layer is created with the same size prior to decoding.  Set to 0 to
            not use an additional hidden layer.

        n_latent_samples: int
            This refers to the number of latent Gaussians to sample
            for creating the inner most encoding.

        decoder_filters: optional list of tuples
            encoder filter configuration stored in a list
            as a tuple (filter_number, filter_size, stride, activation function)

        dropout : bool, optional
            Use dropout or not

        denoising : bool, optional
            Whether or not to apply denoising.  If using denoising, you must feed a
            value for 'corrupt_prob', as returned in the dictionary.  1.0 means no
            corruption is used.  0.0 means every feature is corrupted.  Sensible
            values are between 0.5-0.8.

        img_step : int, optional
            How often to save training images showing the manifold and
            reconstruction.
        save_step : int, optional
            How often to save checkpoints.
        ckpt_name : str, optional
            Checkpoints will be named as this, e.g. 'model.ckpt'
        """
        self.files = files
        self.tf_summary_flag = tf_summary_flag
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_examples = n_examples
        self.crop_shape = crop_shape
        self.crop_factor = crop_factor
        self.encoder_filters = encoder_filters
        self.encoder_pool = encoder_pool
        self.n_fc_hidden = n_fc_hidden
        self.n_latent_samples = n_latent_samples
        self.variational = variational
        self.decoder_filters = decoder_filters
        self.denoising = denoising
        self.dropout = dropout
        self.keep_prob = keep_prob
        self.corrupt_prob = corrupt_prob
        self.activation = activation
        self.img_step = img_step
        self.save_step = save_step
        self.recon_number = recon_number
        self.ckpt_name = ckpt_name

    def train(self):
        current_time = time.strftime('%Y%m%d_%H%M%S', time.gmtime())
        batch = create_input_pipeline(
            files=self.files,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            crop_shape=self.crop_shape,
            crop_factor=self.crop_factor,
            shape=self.input_shape)
        ae = self.create()
        # Create a manifold of our inner most layer to show
        # example reconstructions.  This is one way to see
        # what the "embedding" or "latent space" of the encoder
        # is capable of encoding, though note that this is just
        # a random hyperplane within the latent space, and does not
        # encompass all possible embeddings
        zs = np.random.uniform(
            -1.0, 1.0, [4, self.n_latent_samples]).astype(np.float32)
        zs = utils.make_latent_manifold(zs, self.n_examples)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(ae['cost'])

        # We create a session to use the graph
        sess = tf.Session()
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        # op = sess.graph.get_operations()
        # print("======= Tensors =======")
        # [print(m.values()) for m in op][1]
        # print("=======================")
        
        if self.tf_summary_flag:
            sum_op = tf.summary.merge([
                ae['sum_x'],
                ae['sum_y'],
                ae['sum_loss_x'],
                ae['sum_loss_z'],
                ae['sum_loss_cost']])
            writer = tf.summary.FileWriter("./logs", sess.graph)

        # This will handle our threaded image pipeline
        coord = tf.train.Coordinator()

        # Ensure no more changes to graph
        tf.get_default_graph().finalize()

        # Start up the queues for handling the image pipeline
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        if os.path.exists(self.ckpt_name + '.index') or os.path.exists(self.ckpt_name):
            print("Restoring existing model")
            saver.restore(sess, self.ckpt_name)

        model_img_dir = "data/model_imgs/vae" + current_time
        if not os.path.exists(model_img_dir):
            os.makedirs(model_img_dir)

        # Fit all training data
        t_i = 0
        batch_i = 0
        epoch_i = 0
        cost = 0
        n_files = len(self.files)
        test_xs = sess.run(batch) / 255.0
        test_xs = test_xs[:self.recon_number]
        utils.montage(test_xs, os.path.join(model_img_dir, 'test_xs.png'))
        train_cost = 0
        try:
            while not coord.should_stop() and epoch_i < self.n_epochs:
                batch_i += 1
                batch_xs = sess.run(batch) / 255.0
                if self.tf_summary_flag:
                    train_cost = sess.run([ae['cost'], optimizer, sum_op], feed_dict={
                        ae['x']: batch_xs, ae['train']: True,
                        ae['keep_prob']: self.keep_prob,
                        ae['corrupt_prob']: [self.corrupt_prob]})[0]
                else:
                    train_cost = sess.run([ae['cost'], optimizer], feed_dict={
                        ae['x']: batch_xs, ae['train']: True,
                        ae['keep_prob']: self.keep_prob,
                        ae['corrupt_prob']: [self.corrupt_prob]})[0]

                print("Training: ", batch_i, train_cost)
                cost += train_cost
                if batch_i % n_files == 0:
                    print('epoch:', epoch_i)
                    print('average cost:', cost / batch_i)
                    cost = 0
                    batch_i = 0
                    epoch_i += 1
                if batch_i % self.img_step == 0:
                    # Plot example reconstructions from latent layer
                    recon = sess.run(
                        ae['y'], feed_dict={
                            ae['z']: zs,
                            ae['train']: False,
                            ae['keep_prob']: 1.0,
                            ae['corrupt_prob']: [self.corrupt_prob]})
                    utils.montage(
                        recon.reshape([-1] + self.crop_shape),
                        os.path.join(model_img_dir, 'manifold_%08d.png' % t_i))

                    # Plot example reconstructions
                    recon = sess.run(
                        ae['y'], feed_dict={ae['x']: test_xs,
                                            ae['train']: False,
                                            ae['keep_prob']: 1.0,
                                            ae['corrupt_prob']: [self.corrupt_prob]})
                    print(
                        "batch_i: ", batch_i, ", t_i: ", t_i, 'reconstruction (min, max, mean):',
                        recon.min(), recon.max(), recon.mean())
                    utils.montage(
                        recon.reshape([-1] + self.crop_shape),
                        os.path.join(model_img_dir, 'reconstruction_%08d.png' % t_i))
                    t_i += 1

                if batch_i % self.save_step == 0:
                    # Save the variables to disk.
                    saver.save(
                        sess,
                        self.ckpt_name,
                        global_step=batch_i,
                        write_meta_graph=False)

                sys.stdout.flush()
        except tf.errors.OutOfRangeError:
            print('Done.')
        finally:
            # One of the threads has issued an exception.  So let's tell all the
            # threads to shutdown.
            coord.request_stop()

        # Wait until all threads have finished.
        coord.join(threads)

        # Clean up the session.
        sess.close()

    def create(self):
        """ create function
        Returns
        -------
        model : dict
            {
                'cost': Tensor to optimize.
                'Ws': All weights of the encoder.
                'x': Input Placeholder
                'z': Inner most encoding Tensor (latent features)
                'y': Reconstruction of the Decoder
                'keep_prob': Amount to keep when using Dropout
                'corrupt_prob': Amount to corrupt when using Denoising
                'train': Set to True when training/Applies to Batch Normalization.
            }
        """
        sum_x = None
        sum_y = None
        sum_loss_x = None
        sum_loss_z = None
        sum_loss_cost = None
        input_shape = [None] + self.crop_shape
        x = tf.placeholder(tf.float32, input_shape, 'x')
        if self.tf_summary_flag:
            sum_x = tf.summary.image("x", x)
        phase_train = tf.placeholder(tf.bool, name='phase_train')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        corrupt_prob = tf.placeholder(tf.float32, [1])

        # apply noise if denoising
        x_ = (utils.corrupt(x) * corrupt_prob + x * (1 - corrupt_prob)) if self.denoising else x

        # 2d -> 4d if convolution
        x_tensor = utils.to_tensor(x_)
        current_input = x_tensor

        Ws = []
        shapes = []
        # check if pooling layer is present
        if self.encoder_pool is not None:
            if len(self.encoder_pool) != len(self.encoder_filters):
                print("number of elements for encoder_pool not equal to encoder filters")
                return None
        print("========= ENCODER =========")
        print("input: ", current_input.get_shape().as_list())
        print("encoder_filters: ", self.encoder_filters)
        # Build the encoder
        for layer_i, enc in enumerate(self.encoder_filters):
            # number of filters
            n_filters = enc[0]
            # fileter size
            filter_size = enc[1]
            # stride
            stride = enc[2]
            # activation function
            activation_fn = enc[3]
            # pooling
            pool_filter_size = None
            pool_stride = None
            if self.encoder_pool is not None:
                if self.encoder_pool[layer_i] is not None:
                    pool_fn = self.encoder_pool[layer_i][0]
                    pool_filter_size = self.encoder_pool[layer_i][1]
                    pool_stride = self.encoder_pool[layer_i][2]
            with tf.variable_scope('encoder/{}'.format(layer_i)):
                h, W = utils.conv2d(x=current_input,
                                    n_output=n_filters,
                                    k_h=filter_size,
                                    k_w=filter_size,
                                    d_h=stride, d_w=stride)
                # activation layer
                h = self.activation(batch_norm(h, phase_train, 'bn' + str(layer_i)))
                # add pooling layer if available
                if pool_filter_size is not None and pool_stride is not None:
                    h = pool_fn(
                        h,
                        ksize=[1, pool_filter_size, pool_filter_size, 1],
                        strides=[1, pool_stride, pool_stride, 1],
                        padding='SAME',
                        name='pool' + str(layer_i))

                if self.dropout:
                    h = tf.nn.dropout(h, keep_prob)
                Ws.append(W)
                current_input = h
            print("current_input ", layer_i, ": ", current_input.get_shape().as_list())
            shapes.append(current_input.get_shape().as_list())

        # variational layer
        with tf.variable_scope('variational'):
            if self.variational:
                dims = current_input.get_shape().as_list()
                flattened = utils.flatten(current_input)

                if self.n_fc_hidden:
                    h = utils.linear(flattened, self.n_fc_hidden, name='W_fc')[0]
                    h = self.activation(batch_norm(h, phase_train, 'fc/bn'))
                    if self.dropout:
                        h = tf.nn.dropout(h, keep_prob)
                else:
                    h = flattened

                z_mu = utils.linear(h, self.n_latent_samples, name='mu')[0]
                z_log_sigma = 0.5 * utils.linear(h, self.n_latent_samples, name='log_sigma')[0]

                # Sample from noise distribution p(eps) ~ N(0, 1)
                epsilon = tf.random_normal(
                    tf.stack([tf.shape(x)[0], self.n_latent_samples]))

                # Sample from posterior
                z = z_mu + tf.multiply(epsilon, tf.exp(z_log_sigma))

                if self.n_fc_hidden:
                    h = utils.linear(z, self.n_fc_hidden, name='fc_t')[0]
                    h = self.activation(batch_norm(h, phase_train, 'fc_t/bn'))
                    if self.dropout:
                        h = tf.nn.dropout(h, keep_prob)
                else:
                    h = z

                size = dims[1] * dims[2] * dims[3]
                h = utils.linear(h, size, name='fc_t2')[0]
                current_input = self.activation(batch_norm(h, phase_train, 'fc_t2/bn'))
                if self.dropout:
                    current_input = tf.nn.dropout(current_input, self.keep_prob)

                current_input = tf.reshape(
                    current_input, tf.stack([
                        tf.shape(current_input)[0],
                        dims[1],
                        dims[2],
                        dims[3]]))
                print("current_input variational: ", current_input.get_shape().as_list())
            else:
                z = current_input
        # decoder
        decoder_dimensions = []
        dim_denominator = 0

        # add original image channel number to end of the decode filters
        # so that the image is reconstructed at the tail end of the decoder
        for i in range(len(self.decoder_filters)):
            if i < 1:
                dim_denominator = 1
            else:
                dim_denominator = dim_denominator*2
            # calculate length and width of output shape, so that
            # the original image is reconstucted at the tail end of the decoder
            decoder_dimensions.append(
                (input_shape[-3]//dim_denominator, input_shape[-2]//dim_denominator))
        decoder_dimensions = decoder_dimensions[::-1]
        print("decoder_dimensions: ", decoder_dimensions)
        print("decoder_filters: ", self.decoder_filters)
        input_ch = shapes[-1][3]
        # Decoding layers
        print("========= DECODER =========")
        print("input: ", current_input.get_shape().as_list())
        for layer_i, decoder_dim in enumerate(decoder_dimensions):
            n_output_ch = self.decoder_filters[layer_i][0]
            filter_size = self.decoder_filters[layer_i][1]
            activation_fn = self.decoder_filters[layer_i][2]
            with tf.variable_scope('decoder/{}'.format(layer_i)):
                h, W = utils.deconv2d(
                    x=current_input,
                    n_output_h=decoder_dim[0],
                    n_output_w=decoder_dim[1],
                    n_output_ch=n_output_ch,
                    n_input_ch=input_ch,
                    k_h=filter_size,
                    k_w=filter_size)
                h = activation_fn(batch_norm(h, phase_train, 'dec/bn' + str(layer_i)))
                if self.dropout:
                    h = tf.nn.dropout(h, keep_prob)
                current_input = h
                input_ch = n_output_ch
                shapes.append(current_input.get_shape().as_list())
                print("current_input ", layer_i, ": ", current_input.get_shape().as_list())
        y = current_input

        if self.tf_summary_flag:
            sum_y = tf.summary.image("y", y)

        x_flat = utils.flatten(x)
        y_flat = utils.flatten(y)

        # l2 loss
        loss_x = tf.reduce_sum(tf.squared_difference(x_flat, y_flat), 1)
        if self.tf_summary_flag:
            sum_loss_x = tf.summary.scalar("loss_x", loss_x)

        if self.variational:
            # variational lower bound, kl-divergence
            loss_z = -0.5 * tf.reduce_sum(
                1.0 + 2.0 * z_log_sigma -
                tf.square(z_mu) - tf.exp(2.0 * z_log_sigma), 1)

            if self.tf_summary_flag:
                sum_loss_z = tf.summary.scalar("loss_z", loss_z)
            # add l2 loss
            cost = tf.reduce_mean(loss_x + loss_z)
        else:
            # just optimize l2 loss
            cost = tf.reduce_mean(loss_x)

        if self.tf_summary_flag:
            sum_loss_cost = tf.summary.scalar("cost", cost)

        return {'cost': cost, 'Ws': Ws,
                'x': x, 'z': z, 'y': y,
                'keep_prob': keep_prob,
                'corrupt_prob': corrupt_prob,
                'train': phase_train,
                'sum_x': sum_x,
                'sum_y': sum_y,
                'sum_loss_x': sum_loss_x,
                'sum_loss_z': sum_loss_z,
                'sum_loss_cost': sum_loss_cost}
