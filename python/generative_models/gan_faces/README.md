
# Session 5: Generative Networks
## Assignment: Generative Adversarial Networks and Recurrent Neural Networks

<p class="lead">
<a href="https://www.kadenze.com/courses/creative-applications-of-deep-learning-with-tensorflow/info">Creative Applications of Deep Learning with Google's Tensorflow</a><br />
<a href="http://pkmital.com">Parag K. Mital</a><br />
<a href="https://www.kadenze.com">Kadenze, Inc.</a>
</p>

# Table of Contents

<!-- MarkdownTOC autolink="true" autoanchor="true" bracket="round" -->

- [Overview](#overview)
- [Learning Goals](#learning-goals)
- [Part 1 - Generative Adversarial Networks \(GAN\) / Deep Convolutional GAN \(DCGAN\)](#part-1---generative-adversarial-networks-gan--deep-convolutional-gan-dcgan)
  - [Introduction](#introduction)
  - [Building the Encoder](#building-the-encoder)
  - [Building the Discriminator for the Training Samples](#building-the-discriminator-for-the-training-samples)
  - [Building the Decoder](#building-the-decoder)
  - [Building the Generator](#building-the-generator)
  - [Building the Discriminator for the Generated Samples](#building-the-discriminator-for-the-generated-samples)
  - [GAN Loss Functions](#gan-loss-functions)
  - [Building the Optimizers w/ Regularization](#building-the-optimizers-w-regularization)
  - [Loading a Dataset](#loading-a-dataset)
  - [Training](#training)
  - [Equilibrium](#equilibrium)
- [Part 2 - Variational Auto-Encoding Generative Adversarial Network \(VAEGAN\)](#part-2---variational-auto-encoding-generative-adversarial-network-vaegan)
  - [Batch Normalization](#batch-normalization)
  - [Building the Encoder](#building-the-encoder-1)
  - [Building the Variational Layer](#building-the-variational-layer)
  - [Building the Decoder](#building-the-decoder-1)
  - [Building VAE/GAN Loss Functions](#building-vaegan-loss-functions)
  - [Creating the Optimizers](#creating-the-optimizers)
  - [Loading the Dataset](#loading-the-dataset)
  - [Training](#training-1)
- [Part 3 - Latent-Space Arithmetic](#part-3---latent-space-arithmetic)
  - [Loading the Pre-Trained Model](#loading-the-pre-trained-model)
  - [Exploring the Celeb Net Attributes](#exploring-the-celeb-net-attributes)
  - [Find the Latent Encoding for an Attribute](#find-the-latent-encoding-for-an-attribute)
  - [Latent Feature Arithmetic](#latent-feature-arithmetic)
  - [Extensions](#extensions)
- [Part 4 - Character-Level Language Model](session-5-part-2.ipynb#part-4---character-level-language-model)
- [Part 5 - Pretrained Char-RNN of Donald Trump](session-5-part-2.ipynb#part-5---pretrained-char-rnn-of-donald-trump)
    - [Getting the Trump Data](session-5-part-2.ipynb#getting-the-trump-data)
    - [Basic Text Analysis](session-5-part-2.ipynb#basic-text-analysis)
    - [Loading the Pre-trained Trump Model](session-5-part-2.ipynb#loading-the-pre-trained-trump-model)
    - [Inference: Keeping Track of the State](session-5-part-2.ipynb#inference-keeping-track-of-the-state)
    - [Probabilistic Sampling](session-5-part-2.ipynb#probabilistic-sampling)
    - [Inference: Temperature](session-5-part-2.ipynb#inference-temperature)
    - [Inference: Priming](session-5-part-2.ipynb#inference-priming)
- [Assignment Submission](session-5-part-2.ipynb#assignment-submission)
<!-- /MarkdownTOC -->


<a name="overview"></a>
# Overview

This is certainly the hardest session and will require a lot of time and patience to complete.  Also, many elements of this session may require further investigation, including reading of the original papers and additional resources in order to fully grasp their understanding.  The models we cover are state of the art and I've aimed to give you something between a practical and mathematical understanding of the material, though it is a tricky balance.  I hope for those interested, that you delve deeper into the papers for more understanding.  And for those of you seeking just a practical understanding, that these notebooks will suffice.

This session covered two of the most advanced generative networks: generative adversarial networks and recurrent neural networks.  During the homework, we'll see how these work in more details and try building our own.  I am not asking you train anything in this session as both GANs and RNNs take many days to train.  However, I have provided pre-trained networks which we'll be exploring.   We'll also see how a Variational Autoencoder can be combined with a Generative Adversarial Network to allow you to also encode input data, and I've provided a pre-trained model of this type of model trained on the Celeb Faces dataset.  We'll see what this means in more details below.

After this session, you are also required to submit your final project which can combine any of the materials you have learned so far to produce a short 1 minute clip demonstrating any aspect of the course you want to invesitgate further or combine with anything else you feel like doing.  This is completely open to you and to encourage your peers to share something that demonstrates creative thinking.  Be sure to keep the final project in mind while browsing through this notebook!

<a name="learning-goals"></a>
# Learning Goals

* Learn to build the components of a Generative Adversarial Network and how it is trained
* Learn to combine the Variational Autoencoder with a Generative Adversarial Network
* Learn to use latent space arithmetic with a pre-trained VAE/GAN network
* Learn to build the components of a Character Recurrent Neural Network and how it is trained
* Learn to sample from a pre-trained CharRNN model


```python
# First check the Python version
import sys
if sys.version_info < (3,4):
    print('You are running an older version of Python!\n\n',
          'You should consider updating to Python 3.4.0 or',
          'higher as the libraries built for this course',
          'have only been tested in Python 3.4 and higher.\n')
    print('Try installing the Python 3.5 version of anaconda'
          'and then restart `jupyter notebook`:\n',
          'https://www.continuum.io/downloads\n\n')

# Now get necessary libraries
try:
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.transform import resize
    from skimage import data
    from scipy.misc import imresize
    from scipy.ndimage.filters import gaussian_filter
    import IPython.display as ipyd
    import tensorflow as tf
    from libs import utils, gif, datasets, dataset_utils, nb_utils
    from tensorflow.contrib.layers import batch_norm
except ImportError as e:
    print("Make sure you have started notebook in the same directory",
          "as the provided zip file which includes the 'libs' folder",
          "and the file 'utils.py' inside of it.  You will NOT be able",
          "to complete this assignment unless you restart jupyter",
          "notebook inside the directory created by extracting",
          "the zip file or cloning the github repo.")
    print(e)

# We'll tell matplotlib to inline any drawn figures like so:
%matplotlib inline
plt.style.use('ggplot')
```


```python
# Bit of formatting because I don't like the default inline code style:
from IPython.core.display import HTML
HTML("""<style> .rendered_html code { 
    padding: 2px 4px;
    color: #c7254e;
    background-color: #f9f2f4;
    border-radius: 4px;
} </style>""")
```




<style> .rendered_html code { 
    padding: 2px 4px;
    color: #c7254e;
    background-color: #f9f2f4;
    border-radius: 4px;
} </style>



<a name="part-1---generative-adversarial-networks-gan--deep-convolutional-gan-dcgan"></a>
# Part 1 - Generative Adversarial Networks (GAN) / Deep Convolutional GAN (DCGAN)

<a name="introduction"></a>
## Introduction

Recall from the lecture that a Generative Adversarial Network is two networks, a generator and a discriminator.  The "generator" takes a feature vector and decodes this feature vector to become an image, exactly like the decoder we built in Session 3's Autoencoder.  The discriminator is exactly like the encoder of the Autoencoder, except it can only have 1 value in the final layer.  We use a sigmoid to squash this value between 0 and 1, and then interpret the meaning of it as: 1, the image you gave me was real, or 0, the image you gave me was generated by the generator, it's a FAKE!  So the discriminator is like an encoder which takes an image and then perfoms lie detection.  Are you feeding me lies?  Or is the image real?  

Consider the AE and VAE we trained in Session 3.  The loss function operated partly on the input space.  It said, per pixel, what is the difference between my reconstruction and the input image?  The l2-loss per pixel.  Recall at that time we suggested that this wasn't the best idea because per-pixel differences aren't representative of our own perception of the image.  One way to consider this is if we had the same image, and translated it by a few pixels.  We would not be able to tell the difference, but the per-pixel difference between the two images could be enormously high.

The GAN does not use per-pixel difference.  Instead, it trains a distance function: the discriminator.  The discriminator takes in two images, the real image and the generated one, and learns what a similar image should look like!  That is really the amazing part of this network and has opened up some very exciting potential future directions for unsupervised learning.  Another network that also learns a distance function is known as the siamese network.  We didn't get into this network in this course, but it is commonly used in facial verification, or asserting whether two faces are the same or not.

The GAN network is notoriously a huge pain to train!  For that reason, we won't actually be training it.  Instead, we'll discuss an extension to this basic network called the VAEGAN which uses the VAE we created in Session 3 along with the GAN.  We'll then train that network in Part 2.  For now, let's stick with creating the GAN.

Let's first create the two networks: the discriminator and the generator.  We'll first begin by building a general purpose encoder which we'll use for our discriminator.  Recall that we've already done this in Session 3.  What we want is for the input placeholder to be encoded using a list of dimensions for each of our encoder's layers.  In the case of a convolutional network, our list of dimensions should correspond to the number of output filters.  We also need to specify the kernel heights and widths for each layer's convolutional network.

We'll first need a placeholder.  This will be the "real" image input to the discriminator and the discrimintator will encode this image into a single value, 0 or 1, saying, yes this is real, or no, this is not real.

<h3><font color='red'>TODO! COMPLETE THIS SECTION!</font></h3>


```python
from tensorflow.python.framework import ops
ops.reset_default_graph()
# We'll keep a variable for the size of our image.
n_pixels = 32
n_channels = 3
input_shape = [None, n_pixels, n_pixels, n_channels]

# And then create the input image placeholder
X = tf.placeholder(name='X', shape=input_shape, dtype=tf.float32)
```

<a name="building-the-encoder"></a>
## Building the Encoder

Let's build our encoder just like in Session 3.  We'll create a function which accepts the input placeholder, a list of dimensions describing the number of convolutional filters in each layer, and a list of filter sizes to use for the kernel sizes in each convolutional layer.  We'll also pass in a parameter for which activation function to apply.

<h3><font color='red'>TODO! COMPLETE THIS SECTION!</font></h3>


```python
def encoder(x, channels, filter_sizes, activation=tf.nn.tanh, reuse=None):
    # Set the input to a common variable name, h, for hidden layer
    h = x

    # Now we'll loop over the list of dimensions defining the number
    # of output filters in each layer, and collect each hidden layer
    hs = []
    for layer_i in range(len(channels)):
        
        with tf.variable_scope('layer{}'.format(layer_i+1), reuse=reuse):
            # Convolve using the utility convolution function
            # This requires the number of output filter,
            # and the size of the kernel in `k_h` and `k_w`.
            # By default, this will use a stride of 2, meaning
            # each new layer will be downsampled by 2.
            h, W = utils.conv2d(
                                h, 
                                len(filter_sizes),
                                k_h=3,
                                k_w=3,
                                d_h=2,
                                d_w=2,
                                name=str(layer_i))
            # Now apply the activation function
            h = activation(h)
                                            
            # Store each hidden layer
            hs.append(h)

    # Finally, return the encoding.
    return h, hs
```

<a name="building-the-discriminator-for-the-training-samples"></a>
## Building the Discriminator for the Training Samples

Finally, let's take the output of our encoder, and make sure it has just 1 value by using a fully connected layer.  We can use the `libs/utils` module's, `linear` layer to do this, which will also reshape our 4-dimensional tensor to a 2-dimensional one prior to using the fully connected layer.

<h3><font color='red'>TODO! COMPLETE THIS SECTION!</font></h3>


```python
def discriminator(X,
                  channels=[50, 50, 50, 50],
                  filter_sizes=[4, 4, 4, 4],
                  activation=utils.lrelu,
                  reuse=None):

    # We'll scope these variables to "discriminator_real"
    with tf.variable_scope('discriminator', reuse=reuse):
        # Encode X:
        H, Hs = encoder(X, channels, filter_sizes, activation, reuse)
        
        # Now make one last layer with just 1 output.  We'll
        # have to reshape to 2-d so that we can create a fully
        # connected layer:
        shape = H.get_shape().as_list()
        H = tf.reshape(H, [-1, shape[1] * shape[2] * shape[3]])
        
        # Now we can connect our 2D layer to a single neuron output w/
        # a sigmoid activation:
        D, W = utils.linear(x=H, n_output=1, name='Y', activation=tf.nn.sigmoid, reuse=None)
    return D
```

Now let's create the discriminator for the real training data coming from `X`:


```python
D_real = discriminator(X)
```

And we can see what the network looks like now:


```python
graph = tf.get_default_graph()
nb_utils.show_graph(graph.as_graph_def())
```



            <iframe seamless style="width:800px;height:620px;border:0" srcdoc="
            <script>
              function load() {
                document.getElementById(&quot;graph0.5295216639713634&quot;).pbtxt = 'node {\n  name: &quot;X&quot;\n  op: &quot;Placeholder&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: -1\n        }\n        dim {\n          size: 32\n        }\n        dim {\n          size: 32\n        }\n        dim {\n          size: 3\n        }\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/W/Initializer/random/uniform/shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        tensor_content: &quot;\\003\\000\\000\\000\\003\\000\\000\\000\\003\\000\\000\\000\\004\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/W/Initializer/random/uniform/min&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: -0.30860671401023865\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/W/Initializer/random/uniform/max&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.30860671401023865\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/W/Initializer/random/uniform/RandomUniform&quot;\n  op: &quot;RandomUniform&quot;\n  input: &quot;discriminator/layer1/0/W/Initializer/random/uniform/shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;seed&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;seed2&quot;\n    value {\n      i: 0\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/W/Initializer/random/uniform/sub&quot;\n  op: &quot;Sub&quot;\n  input: &quot;discriminator/layer1/0/W/Initializer/random/uniform/max&quot;\n  input: &quot;discriminator/layer1/0/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/W/Initializer/random/uniform/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer1/0/W/Initializer/random/uniform/RandomUniform&quot;\n  input: &quot;discriminator/layer1/0/W/Initializer/random/uniform/sub&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/W/Initializer/random/uniform&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/layer1/0/W/Initializer/random/uniform/mul&quot;\n  input: &quot;discriminator/layer1/0/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/W&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 3\n        }\n        dim {\n          size: 3\n        }\n        dim {\n          size: 3\n        }\n        dim {\n          size: 4\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/W/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;discriminator/layer1/0/W&quot;\n  input: &quot;discriminator/layer1/0/W/Initializer/random/uniform&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/W/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;discriminator/layer1/0/W&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;X&quot;\n  input: &quot;discriminator/layer1/0/W/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/b/Initializer/Const&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        float_val: 0.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/b&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 4\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/b/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;discriminator/layer1/0/b&quot;\n  input: &quot;discriminator/layer1/0/b/Initializer/Const&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/b/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;discriminator/layer1/0/b&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/b&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/h&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;discriminator/layer1/0/conv&quot;\n  input: &quot;discriminator/layer1/0/b/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/mul/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.6000000238418579\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer1/mul/x&quot;\n  input: &quot;discriminator/layer1/0/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/Abs&quot;\n  op: &quot;Abs&quot;\n  input: &quot;discriminator/layer1/0/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/mul/1/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.4000000059604645\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/mul/1&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer1/mul/1/x&quot;\n  input: &quot;discriminator/layer1/Abs&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/add&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/layer1/mul&quot;\n  input: &quot;discriminator/layer1/mul/1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/W/Initializer/random/uniform/shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        tensor_content: &quot;\\003\\000\\000\\000\\003\\000\\000\\000\\004\\000\\000\\000\\004\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/W/Initializer/random/uniform/min&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: -0.28867512941360474\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/W/Initializer/random/uniform/max&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.28867512941360474\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/W/Initializer/random/uniform/RandomUniform&quot;\n  op: &quot;RandomUniform&quot;\n  input: &quot;discriminator/layer2/1/W/Initializer/random/uniform/shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;seed&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;seed2&quot;\n    value {\n      i: 0\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/W/Initializer/random/uniform/sub&quot;\n  op: &quot;Sub&quot;\n  input: &quot;discriminator/layer2/1/W/Initializer/random/uniform/max&quot;\n  input: &quot;discriminator/layer2/1/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/W/Initializer/random/uniform/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer2/1/W/Initializer/random/uniform/RandomUniform&quot;\n  input: &quot;discriminator/layer2/1/W/Initializer/random/uniform/sub&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/W/Initializer/random/uniform&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/layer2/1/W/Initializer/random/uniform/mul&quot;\n  input: &quot;discriminator/layer2/1/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/W&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 3\n        }\n        dim {\n          size: 3\n        }\n        dim {\n          size: 4\n        }\n        dim {\n          size: 4\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/W/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;discriminator/layer2/1/W&quot;\n  input: &quot;discriminator/layer2/1/W/Initializer/random/uniform&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/W/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;discriminator/layer2/1/W&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;discriminator/layer1/add&quot;\n  input: &quot;discriminator/layer2/1/W/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/b/Initializer/Const&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        float_val: 0.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/b&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 4\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/b/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;discriminator/layer2/1/b&quot;\n  input: &quot;discriminator/layer2/1/b/Initializer/Const&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/b/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;discriminator/layer2/1/b&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/b&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/h&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;discriminator/layer2/1/conv&quot;\n  input: &quot;discriminator/layer2/1/b/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/mul/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.6000000238418579\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer2/mul/x&quot;\n  input: &quot;discriminator/layer2/1/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/Abs&quot;\n  op: &quot;Abs&quot;\n  input: &quot;discriminator/layer2/1/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/mul/1/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.4000000059604645\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/mul/1&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer2/mul/1/x&quot;\n  input: &quot;discriminator/layer2/Abs&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/add&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/layer2/mul&quot;\n  input: &quot;discriminator/layer2/mul/1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/W/Initializer/random/uniform/shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        tensor_content: &quot;\\003\\000\\000\\000\\003\\000\\000\\000\\004\\000\\000\\000\\004\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/W/Initializer/random/uniform/min&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: -0.28867512941360474\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/W/Initializer/random/uniform/max&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.28867512941360474\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/W/Initializer/random/uniform/RandomUniform&quot;\n  op: &quot;RandomUniform&quot;\n  input: &quot;discriminator/layer3/2/W/Initializer/random/uniform/shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;seed&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;seed2&quot;\n    value {\n      i: 0\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/W/Initializer/random/uniform/sub&quot;\n  op: &quot;Sub&quot;\n  input: &quot;discriminator/layer3/2/W/Initializer/random/uniform/max&quot;\n  input: &quot;discriminator/layer3/2/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/W/Initializer/random/uniform/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer3/2/W/Initializer/random/uniform/RandomUniform&quot;\n  input: &quot;discriminator/layer3/2/W/Initializer/random/uniform/sub&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/W/Initializer/random/uniform&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/layer3/2/W/Initializer/random/uniform/mul&quot;\n  input: &quot;discriminator/layer3/2/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/W&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 3\n        }\n        dim {\n          size: 3\n        }\n        dim {\n          size: 4\n        }\n        dim {\n          size: 4\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/W/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;discriminator/layer3/2/W&quot;\n  input: &quot;discriminator/layer3/2/W/Initializer/random/uniform&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/W/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;discriminator/layer3/2/W&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;discriminator/layer2/add&quot;\n  input: &quot;discriminator/layer3/2/W/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/b/Initializer/Const&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        float_val: 0.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/b&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 4\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/b/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;discriminator/layer3/2/b&quot;\n  input: &quot;discriminator/layer3/2/b/Initializer/Const&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/b/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;discriminator/layer3/2/b&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/b&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/h&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;discriminator/layer3/2/conv&quot;\n  input: &quot;discriminator/layer3/2/b/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/mul/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.6000000238418579\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer3/mul/x&quot;\n  input: &quot;discriminator/layer3/2/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/Abs&quot;\n  op: &quot;Abs&quot;\n  input: &quot;discriminator/layer3/2/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/mul/1/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.4000000059604645\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/mul/1&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer3/mul/1/x&quot;\n  input: &quot;discriminator/layer3/Abs&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/add&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/layer3/mul&quot;\n  input: &quot;discriminator/layer3/mul/1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/W/Initializer/random/uniform/shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        tensor_content: &quot;\\003\\000\\000\\000\\003\\000\\000\\000\\004\\000\\000\\000\\004\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/W/Initializer/random/uniform/min&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: -0.28867512941360474\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/W/Initializer/random/uniform/max&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.28867512941360474\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/W/Initializer/random/uniform/RandomUniform&quot;\n  op: &quot;RandomUniform&quot;\n  input: &quot;discriminator/layer4/3/W/Initializer/random/uniform/shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;seed&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;seed2&quot;\n    value {\n      i: 0\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/W/Initializer/random/uniform/sub&quot;\n  op: &quot;Sub&quot;\n  input: &quot;discriminator/layer4/3/W/Initializer/random/uniform/max&quot;\n  input: &quot;discriminator/layer4/3/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/W/Initializer/random/uniform/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer4/3/W/Initializer/random/uniform/RandomUniform&quot;\n  input: &quot;discriminator/layer4/3/W/Initializer/random/uniform/sub&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/W/Initializer/random/uniform&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/layer4/3/W/Initializer/random/uniform/mul&quot;\n  input: &quot;discriminator/layer4/3/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/W&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 3\n        }\n        dim {\n          size: 3\n        }\n        dim {\n          size: 4\n        }\n        dim {\n          size: 4\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/W/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;discriminator/layer4/3/W&quot;\n  input: &quot;discriminator/layer4/3/W/Initializer/random/uniform&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/W/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;discriminator/layer4/3/W&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;discriminator/layer3/add&quot;\n  input: &quot;discriminator/layer4/3/W/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/b/Initializer/Const&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        float_val: 0.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/b&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 4\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/b/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;discriminator/layer4/3/b&quot;\n  input: &quot;discriminator/layer4/3/b/Initializer/Const&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/b/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;discriminator/layer4/3/b&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/b&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/h&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;discriminator/layer4/3/conv&quot;\n  input: &quot;discriminator/layer4/3/b/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/mul/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.6000000238418579\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer4/mul/x&quot;\n  input: &quot;discriminator/layer4/3/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/Abs&quot;\n  op: &quot;Abs&quot;\n  input: &quot;discriminator/layer4/3/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/mul/1/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.4000000059604645\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/mul/1&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer4/mul/1/x&quot;\n  input: &quot;discriminator/layer4/Abs&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/add&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/layer4/mul&quot;\n  input: &quot;discriminator/layer4/mul/1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Reshape/shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 2\n          }\n        }\n        tensor_content: &quot;\\377\\377\\377\\377\\020\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Reshape&quot;\n  op: &quot;Reshape&quot;\n  input: &quot;discriminator/layer4/add&quot;\n  input: &quot;discriminator/Reshape/shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;Tshape&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/W/Initializer/random/uniform/shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 2\n          }\n        }\n        tensor_content: &quot;\\020\\000\\000\\000\\001\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/W/Initializer/random/uniform/min&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: -0.5940885543823242\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/W/Initializer/random/uniform/max&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.5940885543823242\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/W/Initializer/random/uniform/RandomUniform&quot;\n  op: &quot;RandomUniform&quot;\n  input: &quot;discriminator/Y/W/Initializer/random/uniform/shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;seed&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;seed2&quot;\n    value {\n      i: 0\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/W/Initializer/random/uniform/sub&quot;\n  op: &quot;Sub&quot;\n  input: &quot;discriminator/Y/W/Initializer/random/uniform/max&quot;\n  input: &quot;discriminator/Y/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/W/Initializer/random/uniform/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/Y/W/Initializer/random/uniform/RandomUniform&quot;\n  input: &quot;discriminator/Y/W/Initializer/random/uniform/sub&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/W/Initializer/random/uniform&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/Y/W/Initializer/random/uniform/mul&quot;\n  input: &quot;discriminator/Y/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/W&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 16\n        }\n        dim {\n          size: 1\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/W/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;discriminator/Y/W&quot;\n  input: &quot;discriminator/Y/W/Initializer/random/uniform&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/W/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;discriminator/Y/W&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/b/Initializer/Const&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1\n          }\n        }\n        float_val: 0.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/b&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 1\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/b/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;discriminator/Y/b&quot;\n  input: &quot;discriminator/Y/b/Initializer/Const&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/b/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;discriminator/Y/b&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/b&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/MatMul&quot;\n  op: &quot;MatMul&quot;\n  input: &quot;discriminator/Reshape&quot;\n  input: &quot;discriminator/Y/W/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;transpose_a&quot;\n    value {\n      b: false\n    }\n  }\n  attr {\n    key: &quot;transpose_b&quot;\n    value {\n      b: false\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/h&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;discriminator/Y/MatMul&quot;\n  input: &quot;discriminator/Y/b/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/Sigmoid&quot;\n  op: &quot;Sigmoid&quot;\n  input: &quot;discriminator/Y/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\n';
              }
            </script>
            <link rel=&quot;import&quot; href=&quot;https://tensorboard.appspot.com/tf-graph-basic.build.html&quot; onload=load()>
            <div style=&quot;height:600px&quot;>
              <tf-graph-basic id=&quot;graph0.5295216639713634&quot;></tf-graph-basic>
            </div>
        "></iframe>
        


<a name="building-the-decoder"></a>
## Building the Decoder

Now we're ready to build the Generator, or decoding network.  This network takes as input a vector of features and will try to produce an image that looks like our training data.  We'll send this synthesized image to our discriminator which we've just built above.

Let's start by building the input to this network.  We'll need a placeholder for the input features to this network.  We have to be mindful of how many features we have.  The feature vector for the Generator will eventually need to form an image.  What we can do is create a 1-dimensional vector of values for each element in our batch, giving us `[None, n_features]`.  We can then reshape this to a 4-dimensional Tensor so that we can build a decoder network just like in Session 3.

But how do we assign the values from our 1-d feature vector (or 2-d tensor with Batch number of them) to the 3-d shape of an image (or 4-d tensor with Batch number of them)?  We have to go from the number of features in our 1-d feature vector, let's say `n_latent` to `height x width x channels` through a series of convolutional transpose layers.  One way to approach this is think of the reverse process.  Starting from the final decoding of `height x width x channels`, I will use convolution with a stride of 2, so downsample by 2 with each new layer.  So the second to last decoder layer would be, `height // 2 x width // 2 x ?`.  If I look at it like this, I can use the variable `n_pixels` denoting the `height` and `width` to build my decoder, and set the channels to whatever I want.

Let's start with just our 2-d placeholder which will have `None x n_features`, then convert it to a 4-d tensor ready for the decoder part of the network (a.k.a. the generator).


```python
# We'll need some variables first. This will be how many
# channels our generator's feature vector has.  Experiment w/
# this if you are training your own network.
n_code = 16

# And in total how many feature it has, including the spatial dimensions.
n_latent = (n_pixels // 16) * (n_pixels // 16) * n_code

# Let's build the 2-D placeholder, which is the 1-d feature vector for every
# element in our batch.  We'll then reshape this to 4-D for the decoder.
Z = tf.placeholder(name='Z', shape=[None, n_latent], dtype=tf.float32)

# Now we can reshape it to input to the decoder.  Here we have to
# be mindful of the height and width as described before.  We need
# to make the height and width a factor of the final height and width
# that we want.  Since we are using strided convolutions of 2, then
# we can say with 4 layers, that first decoder's layer should be:
# n_pixels / 2 / 2 / 2 / 2, or n_pixels / 16:
Z_tensor = tf.reshape(Z, [-1, n_pixels // 16, n_pixels // 16, n_code])
```

Now we'll build the decoder in much the same way as we built our encoder.  And exactly as we've done in Session 3!  This requires one additional parameter "channels" which is how many output filters we want for each net layer.  We'll interpret the `dimensions` as the height and width of the tensor in each new layer, the `channels` is how many output filters we want for each net layer, and the `filter_sizes` is the size of the filters used for convolution.  We'll default to using a stride of two which will downsample each layer.  We're also going to collect each hidden layer `h` in a list.  We'll end up needing this for Part 2 when we combine the variational autoencoder w/ the generative adversarial network.


```python
def decoder(z, dimensions, channels, filter_sizes,
            activation=tf.nn.relu, reuse=None):
    h = z
    hs = []
    for layer_i in range(len(dimensions)):
        with tf.variable_scope('layer{}'.format(layer_i+1), reuse=reuse):
            h, W = utils.deconv2d(x=h,
                               n_output_h=dimensions[layer_i],
                               n_output_w=dimensions[layer_i],
                               n_output_ch=channels[layer_i],
                               k_h=filter_sizes[layer_i],
                               k_w=filter_sizes[layer_i],
                               reuse=reuse)
            h = activation(h)
            hs.append(h)
    return h, hs
```

<a name="building-the-generator"></a>
## Building the Generator

Now we're ready to use our decoder to take in a vector of features and generate something that looks like our training images. We have to ensure that the last layer produces the same output shape as the discriminator's input.  E.g. we used a `[None, 64, 64, 3]` input to the discriminator, so our generator needs to also output `[None, 64, 64, 3]` tensors.  In other words, we have to ensure the last element in our `dimensions` list is 64, and the last element in our `channels` list is 3.


```python
# Explore these parameters.
def generator(Z,
              dimensions=[n_pixels//8, n_pixels//4, n_pixels//2, n_pixels],
              channels=[50, 50, 50, n_channels],
              filter_sizes=[4, 4, 4, 4],
              activation=utils.lrelu):

    with tf.variable_scope('generator'):
        G, Hs = decoder(Z_tensor, dimensions, channels, filter_sizes, activation)

    return G
```

Now let's call the `generator` function with our input placeholder `Z`.  This will take our feature vector and generate something in the shape of an image.


```python
G = generator(Z)
```


```python
graph = tf.get_default_graph()
nb_utils.show_graph(graph.as_graph_def())
```



            <iframe seamless style="width:800px;height:620px;border:0" srcdoc="
            <script>
              function load() {
                document.getElementById(&quot;graph0.9779902055767046&quot;).pbtxt = 'node {\n  name: &quot;X&quot;\n  op: &quot;Placeholder&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: -1\n        }\n        dim {\n          size: 32\n        }\n        dim {\n          size: 32\n        }\n        dim {\n          size: 3\n        }\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/W/Initializer/random/uniform/shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        tensor_content: &quot;\\003\\000\\000\\000\\003\\000\\000\\000\\003\\000\\000\\000\\004\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/W/Initializer/random/uniform/min&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: -0.30860671401023865\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/W/Initializer/random/uniform/max&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.30860671401023865\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/W/Initializer/random/uniform/RandomUniform&quot;\n  op: &quot;RandomUniform&quot;\n  input: &quot;discriminator/layer1/0/W/Initializer/random/uniform/shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;seed&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;seed2&quot;\n    value {\n      i: 0\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/W/Initializer/random/uniform/sub&quot;\n  op: &quot;Sub&quot;\n  input: &quot;discriminator/layer1/0/W/Initializer/random/uniform/max&quot;\n  input: &quot;discriminator/layer1/0/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/W/Initializer/random/uniform/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer1/0/W/Initializer/random/uniform/RandomUniform&quot;\n  input: &quot;discriminator/layer1/0/W/Initializer/random/uniform/sub&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/W/Initializer/random/uniform&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/layer1/0/W/Initializer/random/uniform/mul&quot;\n  input: &quot;discriminator/layer1/0/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/W&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 3\n        }\n        dim {\n          size: 3\n        }\n        dim {\n          size: 3\n        }\n        dim {\n          size: 4\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/W/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;discriminator/layer1/0/W&quot;\n  input: &quot;discriminator/layer1/0/W/Initializer/random/uniform&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/W/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;discriminator/layer1/0/W&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;X&quot;\n  input: &quot;discriminator/layer1/0/W/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/b/Initializer/Const&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        float_val: 0.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/b&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 4\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/b/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;discriminator/layer1/0/b&quot;\n  input: &quot;discriminator/layer1/0/b/Initializer/Const&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/b/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;discriminator/layer1/0/b&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/b&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/h&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;discriminator/layer1/0/conv&quot;\n  input: &quot;discriminator/layer1/0/b/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/mul/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.6000000238418579\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer1/mul/x&quot;\n  input: &quot;discriminator/layer1/0/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/Abs&quot;\n  op: &quot;Abs&quot;\n  input: &quot;discriminator/layer1/0/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/mul/1/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.4000000059604645\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/mul/1&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer1/mul/1/x&quot;\n  input: &quot;discriminator/layer1/Abs&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/add&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/layer1/mul&quot;\n  input: &quot;discriminator/layer1/mul/1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/W/Initializer/random/uniform/shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        tensor_content: &quot;\\003\\000\\000\\000\\003\\000\\000\\000\\004\\000\\000\\000\\004\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/W/Initializer/random/uniform/min&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: -0.28867512941360474\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/W/Initializer/random/uniform/max&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.28867512941360474\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/W/Initializer/random/uniform/RandomUniform&quot;\n  op: &quot;RandomUniform&quot;\n  input: &quot;discriminator/layer2/1/W/Initializer/random/uniform/shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;seed&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;seed2&quot;\n    value {\n      i: 0\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/W/Initializer/random/uniform/sub&quot;\n  op: &quot;Sub&quot;\n  input: &quot;discriminator/layer2/1/W/Initializer/random/uniform/max&quot;\n  input: &quot;discriminator/layer2/1/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/W/Initializer/random/uniform/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer2/1/W/Initializer/random/uniform/RandomUniform&quot;\n  input: &quot;discriminator/layer2/1/W/Initializer/random/uniform/sub&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/W/Initializer/random/uniform&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/layer2/1/W/Initializer/random/uniform/mul&quot;\n  input: &quot;discriminator/layer2/1/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/W&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 3\n        }\n        dim {\n          size: 3\n        }\n        dim {\n          size: 4\n        }\n        dim {\n          size: 4\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/W/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;discriminator/layer2/1/W&quot;\n  input: &quot;discriminator/layer2/1/W/Initializer/random/uniform&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/W/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;discriminator/layer2/1/W&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;discriminator/layer1/add&quot;\n  input: &quot;discriminator/layer2/1/W/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/b/Initializer/Const&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        float_val: 0.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/b&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 4\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/b/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;discriminator/layer2/1/b&quot;\n  input: &quot;discriminator/layer2/1/b/Initializer/Const&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/b/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;discriminator/layer2/1/b&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/b&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/h&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;discriminator/layer2/1/conv&quot;\n  input: &quot;discriminator/layer2/1/b/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/mul/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.6000000238418579\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer2/mul/x&quot;\n  input: &quot;discriminator/layer2/1/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/Abs&quot;\n  op: &quot;Abs&quot;\n  input: &quot;discriminator/layer2/1/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/mul/1/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.4000000059604645\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/mul/1&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer2/mul/1/x&quot;\n  input: &quot;discriminator/layer2/Abs&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/add&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/layer2/mul&quot;\n  input: &quot;discriminator/layer2/mul/1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/W/Initializer/random/uniform/shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        tensor_content: &quot;\\003\\000\\000\\000\\003\\000\\000\\000\\004\\000\\000\\000\\004\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/W/Initializer/random/uniform/min&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: -0.28867512941360474\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/W/Initializer/random/uniform/max&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.28867512941360474\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/W/Initializer/random/uniform/RandomUniform&quot;\n  op: &quot;RandomUniform&quot;\n  input: &quot;discriminator/layer3/2/W/Initializer/random/uniform/shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;seed&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;seed2&quot;\n    value {\n      i: 0\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/W/Initializer/random/uniform/sub&quot;\n  op: &quot;Sub&quot;\n  input: &quot;discriminator/layer3/2/W/Initializer/random/uniform/max&quot;\n  input: &quot;discriminator/layer3/2/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/W/Initializer/random/uniform/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer3/2/W/Initializer/random/uniform/RandomUniform&quot;\n  input: &quot;discriminator/layer3/2/W/Initializer/random/uniform/sub&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/W/Initializer/random/uniform&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/layer3/2/W/Initializer/random/uniform/mul&quot;\n  input: &quot;discriminator/layer3/2/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/W&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 3\n        }\n        dim {\n          size: 3\n        }\n        dim {\n          size: 4\n        }\n        dim {\n          size: 4\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/W/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;discriminator/layer3/2/W&quot;\n  input: &quot;discriminator/layer3/2/W/Initializer/random/uniform&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/W/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;discriminator/layer3/2/W&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;discriminator/layer2/add&quot;\n  input: &quot;discriminator/layer3/2/W/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/b/Initializer/Const&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        float_val: 0.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/b&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 4\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/b/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;discriminator/layer3/2/b&quot;\n  input: &quot;discriminator/layer3/2/b/Initializer/Const&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/b/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;discriminator/layer3/2/b&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/b&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/h&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;discriminator/layer3/2/conv&quot;\n  input: &quot;discriminator/layer3/2/b/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/mul/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.6000000238418579\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer3/mul/x&quot;\n  input: &quot;discriminator/layer3/2/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/Abs&quot;\n  op: &quot;Abs&quot;\n  input: &quot;discriminator/layer3/2/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/mul/1/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.4000000059604645\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/mul/1&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer3/mul/1/x&quot;\n  input: &quot;discriminator/layer3/Abs&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/add&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/layer3/mul&quot;\n  input: &quot;discriminator/layer3/mul/1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/W/Initializer/random/uniform/shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        tensor_content: &quot;\\003\\000\\000\\000\\003\\000\\000\\000\\004\\000\\000\\000\\004\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/W/Initializer/random/uniform/min&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: -0.28867512941360474\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/W/Initializer/random/uniform/max&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.28867512941360474\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/W/Initializer/random/uniform/RandomUniform&quot;\n  op: &quot;RandomUniform&quot;\n  input: &quot;discriminator/layer4/3/W/Initializer/random/uniform/shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;seed&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;seed2&quot;\n    value {\n      i: 0\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/W/Initializer/random/uniform/sub&quot;\n  op: &quot;Sub&quot;\n  input: &quot;discriminator/layer4/3/W/Initializer/random/uniform/max&quot;\n  input: &quot;discriminator/layer4/3/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/W/Initializer/random/uniform/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer4/3/W/Initializer/random/uniform/RandomUniform&quot;\n  input: &quot;discriminator/layer4/3/W/Initializer/random/uniform/sub&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/W/Initializer/random/uniform&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/layer4/3/W/Initializer/random/uniform/mul&quot;\n  input: &quot;discriminator/layer4/3/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/W&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 3\n        }\n        dim {\n          size: 3\n        }\n        dim {\n          size: 4\n        }\n        dim {\n          size: 4\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/W/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;discriminator/layer4/3/W&quot;\n  input: &quot;discriminator/layer4/3/W/Initializer/random/uniform&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/W/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;discriminator/layer4/3/W&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;discriminator/layer3/add&quot;\n  input: &quot;discriminator/layer4/3/W/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/b/Initializer/Const&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        float_val: 0.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/b&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 4\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/b/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;discriminator/layer4/3/b&quot;\n  input: &quot;discriminator/layer4/3/b/Initializer/Const&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/b/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;discriminator/layer4/3/b&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/b&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/h&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;discriminator/layer4/3/conv&quot;\n  input: &quot;discriminator/layer4/3/b/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/mul/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.6000000238418579\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer4/mul/x&quot;\n  input: &quot;discriminator/layer4/3/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/Abs&quot;\n  op: &quot;Abs&quot;\n  input: &quot;discriminator/layer4/3/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/mul/1/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.4000000059604645\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/mul/1&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer4/mul/1/x&quot;\n  input: &quot;discriminator/layer4/Abs&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/add&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/layer4/mul&quot;\n  input: &quot;discriminator/layer4/mul/1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Reshape/shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 2\n          }\n        }\n        tensor_content: &quot;\\377\\377\\377\\377\\020\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Reshape&quot;\n  op: &quot;Reshape&quot;\n  input: &quot;discriminator/layer4/add&quot;\n  input: &quot;discriminator/Reshape/shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;Tshape&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/W/Initializer/random/uniform/shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 2\n          }\n        }\n        tensor_content: &quot;\\020\\000\\000\\000\\001\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/W/Initializer/random/uniform/min&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: -0.5940885543823242\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/W/Initializer/random/uniform/max&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.5940885543823242\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/W/Initializer/random/uniform/RandomUniform&quot;\n  op: &quot;RandomUniform&quot;\n  input: &quot;discriminator/Y/W/Initializer/random/uniform/shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;seed&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;seed2&quot;\n    value {\n      i: 0\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/W/Initializer/random/uniform/sub&quot;\n  op: &quot;Sub&quot;\n  input: &quot;discriminator/Y/W/Initializer/random/uniform/max&quot;\n  input: &quot;discriminator/Y/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/W/Initializer/random/uniform/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/Y/W/Initializer/random/uniform/RandomUniform&quot;\n  input: &quot;discriminator/Y/W/Initializer/random/uniform/sub&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/W/Initializer/random/uniform&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/Y/W/Initializer/random/uniform/mul&quot;\n  input: &quot;discriminator/Y/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/W&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 16\n        }\n        dim {\n          size: 1\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/W/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;discriminator/Y/W&quot;\n  input: &quot;discriminator/Y/W/Initializer/random/uniform&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/W/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;discriminator/Y/W&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/b/Initializer/Const&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1\n          }\n        }\n        float_val: 0.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/b&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 1\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/b/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;discriminator/Y/b&quot;\n  input: &quot;discriminator/Y/b/Initializer/Const&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/b/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;discriminator/Y/b&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/b&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/MatMul&quot;\n  op: &quot;MatMul&quot;\n  input: &quot;discriminator/Reshape&quot;\n  input: &quot;discriminator/Y/W/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;transpose_a&quot;\n    value {\n      b: false\n    }\n  }\n  attr {\n    key: &quot;transpose_b&quot;\n    value {\n      b: false\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/h&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;discriminator/Y/MatMul&quot;\n  input: &quot;discriminator/Y/b/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/Sigmoid&quot;\n  op: &quot;Sigmoid&quot;\n  input: &quot;discriminator/Y/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;Z&quot;\n  op: &quot;Placeholder&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: -1\n        }\n        dim {\n          size: 64\n        }\n      }\n    }\n  }\n}\nnode {\n  name: &quot;Reshape/shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        tensor_content: &quot;\\377\\377\\377\\377\\002\\000\\000\\000\\002\\000\\000\\000\\020\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;Reshape&quot;\n  op: &quot;Reshape&quot;\n  input: &quot;Z&quot;\n  input: &quot;Reshape/shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;Tshape&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/W/Initializer/random/uniform/shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer1/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        tensor_content: &quot;\\004\\000\\000\\000\\004\\000\\000\\0002\\000\\000\\000\\020\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/W/Initializer/random/uniform/min&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer1/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: -0.07537783682346344\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/W/Initializer/random/uniform/max&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer1/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.07537783682346344\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/W/Initializer/random/uniform/RandomUniform&quot;\n  op: &quot;RandomUniform&quot;\n  input: &quot;generator/layer1/deconv2d/W/Initializer/random/uniform/shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer1/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;seed&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;seed2&quot;\n    value {\n      i: 0\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/W/Initializer/random/uniform/sub&quot;\n  op: &quot;Sub&quot;\n  input: &quot;generator/layer1/deconv2d/W/Initializer/random/uniform/max&quot;\n  input: &quot;generator/layer1/deconv2d/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer1/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/W/Initializer/random/uniform/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;generator/layer1/deconv2d/W/Initializer/random/uniform/RandomUniform&quot;\n  input: &quot;generator/layer1/deconv2d/W/Initializer/random/uniform/sub&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer1/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/W/Initializer/random/uniform&quot;\n  op: &quot;Add&quot;\n  input: &quot;generator/layer1/deconv2d/W/Initializer/random/uniform/mul&quot;\n  input: &quot;generator/layer1/deconv2d/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer1/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/W&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer1/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 4\n        }\n        dim {\n          size: 4\n        }\n        dim {\n          size: 50\n        }\n        dim {\n          size: 16\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/W/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;generator/layer1/deconv2d/W&quot;\n  input: &quot;generator/layer1/deconv2d/W/Initializer/random/uniform&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer1/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/W/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;generator/layer1/deconv2d/W&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer1/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/Shape&quot;\n  op: &quot;Shape&quot;\n  input: &quot;Reshape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;out_type&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/strided/slice/stack&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 1\n          }\n        }\n        int_val: 0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/strided/slice/stack_1&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 1\n          }\n        }\n        int_val: 1\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/strided/slice/stack_2&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 1\n          }\n        }\n        int_val: 1\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/strided/slice&quot;\n  op: &quot;StridedSlice&quot;\n  input: &quot;generator/layer1/deconv2d/Shape&quot;\n  input: &quot;generator/layer1/deconv2d/strided/slice/stack&quot;\n  input: &quot;generator/layer1/deconv2d/strided/slice/stack_1&quot;\n  input: &quot;generator/layer1/deconv2d/strided/slice/stack_2&quot;\n  attr {\n    key: &quot;Index&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;begin_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;ellipsis_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;end_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;new_axis_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;shrink_axis_mask&quot;\n    value {\n      i: 1\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/stack/1&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 4\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/stack/2&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 4\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/stack/3&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 50\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/stack&quot;\n  op: &quot;Pack&quot;\n  input: &quot;generator/layer1/deconv2d/strided/slice&quot;\n  input: &quot;generator/layer1/deconv2d/stack/1&quot;\n  input: &quot;generator/layer1/deconv2d/stack/2&quot;\n  input: &quot;generator/layer1/deconv2d/stack/3&quot;\n  attr {\n    key: &quot;N&quot;\n    value {\n      i: 4\n    }\n  }\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;axis&quot;\n    value {\n      i: 0\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/conv/t&quot;\n  op: &quot;Conv2DBackpropInput&quot;\n  input: &quot;generator/layer1/deconv2d/stack&quot;\n  input: &quot;generator/layer1/deconv2d/W/read&quot;\n  input: &quot;Reshape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/b/Initializer/Const&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer1/deconv2d/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 50\n          }\n        }\n        float_val: 0.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/b&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer1/deconv2d/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 50\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/b/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;generator/layer1/deconv2d/b&quot;\n  input: &quot;generator/layer1/deconv2d/b/Initializer/Const&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer1/deconv2d/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/b/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;generator/layer1/deconv2d/b&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer1/deconv2d/b&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/h&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;generator/layer1/deconv2d/conv/t&quot;\n  input: &quot;generator/layer1/deconv2d/b/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/mul/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.6000000238418579\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;generator/layer1/mul/x&quot;\n  input: &quot;generator/layer1/deconv2d/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/Abs&quot;\n  op: &quot;Abs&quot;\n  input: &quot;generator/layer1/deconv2d/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/mul/1/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.4000000059604645\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/mul/1&quot;\n  op: &quot;Mul&quot;\n  input: &quot;generator/layer1/mul/1/x&quot;\n  input: &quot;generator/layer1/Abs&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/add&quot;\n  op: &quot;Add&quot;\n  input: &quot;generator/layer1/mul&quot;\n  input: &quot;generator/layer1/mul/1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/W/Initializer/random/uniform/shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer2/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        tensor_content: &quot;\\004\\000\\000\\000\\004\\000\\000\\0002\\000\\000\\0002\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/W/Initializer/random/uniform/min&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer2/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: -0.06123724207282066\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/W/Initializer/random/uniform/max&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer2/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.06123724207282066\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/W/Initializer/random/uniform/RandomUniform&quot;\n  op: &quot;RandomUniform&quot;\n  input: &quot;generator/layer2/deconv2d/W/Initializer/random/uniform/shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer2/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;seed&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;seed2&quot;\n    value {\n      i: 0\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/W/Initializer/random/uniform/sub&quot;\n  op: &quot;Sub&quot;\n  input: &quot;generator/layer2/deconv2d/W/Initializer/random/uniform/max&quot;\n  input: &quot;generator/layer2/deconv2d/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer2/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/W/Initializer/random/uniform/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;generator/layer2/deconv2d/W/Initializer/random/uniform/RandomUniform&quot;\n  input: &quot;generator/layer2/deconv2d/W/Initializer/random/uniform/sub&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer2/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/W/Initializer/random/uniform&quot;\n  op: &quot;Add&quot;\n  input: &quot;generator/layer2/deconv2d/W/Initializer/random/uniform/mul&quot;\n  input: &quot;generator/layer2/deconv2d/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer2/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/W&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer2/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 4\n        }\n        dim {\n          size: 4\n        }\n        dim {\n          size: 50\n        }\n        dim {\n          size: 50\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/W/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;generator/layer2/deconv2d/W&quot;\n  input: &quot;generator/layer2/deconv2d/W/Initializer/random/uniform&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer2/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/W/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;generator/layer2/deconv2d/W&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer2/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/Shape&quot;\n  op: &quot;Shape&quot;\n  input: &quot;generator/layer1/add&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;out_type&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/strided/slice/stack&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 1\n          }\n        }\n        int_val: 0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/strided/slice/stack_1&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 1\n          }\n        }\n        int_val: 1\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/strided/slice/stack_2&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 1\n          }\n        }\n        int_val: 1\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/strided/slice&quot;\n  op: &quot;StridedSlice&quot;\n  input: &quot;generator/layer2/deconv2d/Shape&quot;\n  input: &quot;generator/layer2/deconv2d/strided/slice/stack&quot;\n  input: &quot;generator/layer2/deconv2d/strided/slice/stack_1&quot;\n  input: &quot;generator/layer2/deconv2d/strided/slice/stack_2&quot;\n  attr {\n    key: &quot;Index&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;begin_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;ellipsis_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;end_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;new_axis_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;shrink_axis_mask&quot;\n    value {\n      i: 1\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/stack/1&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 8\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/stack/2&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 8\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/stack/3&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 50\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/stack&quot;\n  op: &quot;Pack&quot;\n  input: &quot;generator/layer2/deconv2d/strided/slice&quot;\n  input: &quot;generator/layer2/deconv2d/stack/1&quot;\n  input: &quot;generator/layer2/deconv2d/stack/2&quot;\n  input: &quot;generator/layer2/deconv2d/stack/3&quot;\n  attr {\n    key: &quot;N&quot;\n    value {\n      i: 4\n    }\n  }\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;axis&quot;\n    value {\n      i: 0\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/conv/t&quot;\n  op: &quot;Conv2DBackpropInput&quot;\n  input: &quot;generator/layer2/deconv2d/stack&quot;\n  input: &quot;generator/layer2/deconv2d/W/read&quot;\n  input: &quot;generator/layer1/add&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/b/Initializer/Const&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer2/deconv2d/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 50\n          }\n        }\n        float_val: 0.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/b&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer2/deconv2d/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 50\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/b/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;generator/layer2/deconv2d/b&quot;\n  input: &quot;generator/layer2/deconv2d/b/Initializer/Const&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer2/deconv2d/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/b/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;generator/layer2/deconv2d/b&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer2/deconv2d/b&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/h&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;generator/layer2/deconv2d/conv/t&quot;\n  input: &quot;generator/layer2/deconv2d/b/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/mul/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.6000000238418579\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;generator/layer2/mul/x&quot;\n  input: &quot;generator/layer2/deconv2d/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/Abs&quot;\n  op: &quot;Abs&quot;\n  input: &quot;generator/layer2/deconv2d/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/mul/1/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.4000000059604645\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/mul/1&quot;\n  op: &quot;Mul&quot;\n  input: &quot;generator/layer2/mul/1/x&quot;\n  input: &quot;generator/layer2/Abs&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/add&quot;\n  op: &quot;Add&quot;\n  input: &quot;generator/layer2/mul&quot;\n  input: &quot;generator/layer2/mul/1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/W/Initializer/random/uniform/shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer3/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        tensor_content: &quot;\\004\\000\\000\\000\\004\\000\\000\\0002\\000\\000\\0002\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/W/Initializer/random/uniform/min&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer3/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: -0.06123724207282066\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/W/Initializer/random/uniform/max&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer3/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.06123724207282066\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/W/Initializer/random/uniform/RandomUniform&quot;\n  op: &quot;RandomUniform&quot;\n  input: &quot;generator/layer3/deconv2d/W/Initializer/random/uniform/shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer3/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;seed&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;seed2&quot;\n    value {\n      i: 0\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/W/Initializer/random/uniform/sub&quot;\n  op: &quot;Sub&quot;\n  input: &quot;generator/layer3/deconv2d/W/Initializer/random/uniform/max&quot;\n  input: &quot;generator/layer3/deconv2d/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer3/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/W/Initializer/random/uniform/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;generator/layer3/deconv2d/W/Initializer/random/uniform/RandomUniform&quot;\n  input: &quot;generator/layer3/deconv2d/W/Initializer/random/uniform/sub&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer3/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/W/Initializer/random/uniform&quot;\n  op: &quot;Add&quot;\n  input: &quot;generator/layer3/deconv2d/W/Initializer/random/uniform/mul&quot;\n  input: &quot;generator/layer3/deconv2d/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer3/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/W&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer3/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 4\n        }\n        dim {\n          size: 4\n        }\n        dim {\n          size: 50\n        }\n        dim {\n          size: 50\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/W/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;generator/layer3/deconv2d/W&quot;\n  input: &quot;generator/layer3/deconv2d/W/Initializer/random/uniform&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer3/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/W/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;generator/layer3/deconv2d/W&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer3/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/Shape&quot;\n  op: &quot;Shape&quot;\n  input: &quot;generator/layer2/add&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;out_type&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/strided/slice/stack&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 1\n          }\n        }\n        int_val: 0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/strided/slice/stack_1&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 1\n          }\n        }\n        int_val: 1\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/strided/slice/stack_2&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 1\n          }\n        }\n        int_val: 1\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/strided/slice&quot;\n  op: &quot;StridedSlice&quot;\n  input: &quot;generator/layer3/deconv2d/Shape&quot;\n  input: &quot;generator/layer3/deconv2d/strided/slice/stack&quot;\n  input: &quot;generator/layer3/deconv2d/strided/slice/stack_1&quot;\n  input: &quot;generator/layer3/deconv2d/strided/slice/stack_2&quot;\n  attr {\n    key: &quot;Index&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;begin_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;ellipsis_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;end_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;new_axis_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;shrink_axis_mask&quot;\n    value {\n      i: 1\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/stack/1&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 16\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/stack/2&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 16\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/stack/3&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 50\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/stack&quot;\n  op: &quot;Pack&quot;\n  input: &quot;generator/layer3/deconv2d/strided/slice&quot;\n  input: &quot;generator/layer3/deconv2d/stack/1&quot;\n  input: &quot;generator/layer3/deconv2d/stack/2&quot;\n  input: &quot;generator/layer3/deconv2d/stack/3&quot;\n  attr {\n    key: &quot;N&quot;\n    value {\n      i: 4\n    }\n  }\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;axis&quot;\n    value {\n      i: 0\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/conv/t&quot;\n  op: &quot;Conv2DBackpropInput&quot;\n  input: &quot;generator/layer3/deconv2d/stack&quot;\n  input: &quot;generator/layer3/deconv2d/W/read&quot;\n  input: &quot;generator/layer2/add&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/b/Initializer/Const&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer3/deconv2d/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 50\n          }\n        }\n        float_val: 0.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/b&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer3/deconv2d/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 50\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/b/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;generator/layer3/deconv2d/b&quot;\n  input: &quot;generator/layer3/deconv2d/b/Initializer/Const&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer3/deconv2d/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/b/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;generator/layer3/deconv2d/b&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer3/deconv2d/b&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/h&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;generator/layer3/deconv2d/conv/t&quot;\n  input: &quot;generator/layer3/deconv2d/b/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/mul/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.6000000238418579\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;generator/layer3/mul/x&quot;\n  input: &quot;generator/layer3/deconv2d/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/Abs&quot;\n  op: &quot;Abs&quot;\n  input: &quot;generator/layer3/deconv2d/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/mul/1/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.4000000059604645\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/mul/1&quot;\n  op: &quot;Mul&quot;\n  input: &quot;generator/layer3/mul/1/x&quot;\n  input: &quot;generator/layer3/Abs&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/add&quot;\n  op: &quot;Add&quot;\n  input: &quot;generator/layer3/mul&quot;\n  input: &quot;generator/layer3/mul/1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/W/Initializer/random/uniform/shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer4/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        tensor_content: &quot;\\004\\000\\000\\000\\004\\000\\000\\000\\003\\000\\000\\0002\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/W/Initializer/random/uniform/min&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer4/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: -0.08411582559347153\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/W/Initializer/random/uniform/max&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer4/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.08411582559347153\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/W/Initializer/random/uniform/RandomUniform&quot;\n  op: &quot;RandomUniform&quot;\n  input: &quot;generator/layer4/deconv2d/W/Initializer/random/uniform/shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer4/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;seed&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;seed2&quot;\n    value {\n      i: 0\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/W/Initializer/random/uniform/sub&quot;\n  op: &quot;Sub&quot;\n  input: &quot;generator/layer4/deconv2d/W/Initializer/random/uniform/max&quot;\n  input: &quot;generator/layer4/deconv2d/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer4/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/W/Initializer/random/uniform/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;generator/layer4/deconv2d/W/Initializer/random/uniform/RandomUniform&quot;\n  input: &quot;generator/layer4/deconv2d/W/Initializer/random/uniform/sub&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer4/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/W/Initializer/random/uniform&quot;\n  op: &quot;Add&quot;\n  input: &quot;generator/layer4/deconv2d/W/Initializer/random/uniform/mul&quot;\n  input: &quot;generator/layer4/deconv2d/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer4/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/W&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer4/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 4\n        }\n        dim {\n          size: 4\n        }\n        dim {\n          size: 3\n        }\n        dim {\n          size: 50\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/W/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;generator/layer4/deconv2d/W&quot;\n  input: &quot;generator/layer4/deconv2d/W/Initializer/random/uniform&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer4/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/W/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;generator/layer4/deconv2d/W&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer4/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/Shape&quot;\n  op: &quot;Shape&quot;\n  input: &quot;generator/layer3/add&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;out_type&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/strided/slice/stack&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 1\n          }\n        }\n        int_val: 0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/strided/slice/stack_1&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 1\n          }\n        }\n        int_val: 1\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/strided/slice/stack_2&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 1\n          }\n        }\n        int_val: 1\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/strided/slice&quot;\n  op: &quot;StridedSlice&quot;\n  input: &quot;generator/layer4/deconv2d/Shape&quot;\n  input: &quot;generator/layer4/deconv2d/strided/slice/stack&quot;\n  input: &quot;generator/layer4/deconv2d/strided/slice/stack_1&quot;\n  input: &quot;generator/layer4/deconv2d/strided/slice/stack_2&quot;\n  attr {\n    key: &quot;Index&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;begin_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;ellipsis_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;end_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;new_axis_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;shrink_axis_mask&quot;\n    value {\n      i: 1\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/stack/1&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 32\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/stack/2&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 32\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/stack/3&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 3\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/stack&quot;\n  op: &quot;Pack&quot;\n  input: &quot;generator/layer4/deconv2d/strided/slice&quot;\n  input: &quot;generator/layer4/deconv2d/stack/1&quot;\n  input: &quot;generator/layer4/deconv2d/stack/2&quot;\n  input: &quot;generator/layer4/deconv2d/stack/3&quot;\n  attr {\n    key: &quot;N&quot;\n    value {\n      i: 4\n    }\n  }\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;axis&quot;\n    value {\n      i: 0\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/conv/t&quot;\n  op: &quot;Conv2DBackpropInput&quot;\n  input: &quot;generator/layer4/deconv2d/stack&quot;\n  input: &quot;generator/layer4/deconv2d/W/read&quot;\n  input: &quot;generator/layer3/add&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/b/Initializer/Const&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer4/deconv2d/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 3\n          }\n        }\n        float_val: 0.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/b&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer4/deconv2d/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 3\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/b/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;generator/layer4/deconv2d/b&quot;\n  input: &quot;generator/layer4/deconv2d/b/Initializer/Const&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer4/deconv2d/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/b/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;generator/layer4/deconv2d/b&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer4/deconv2d/b&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/h&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;generator/layer4/deconv2d/conv/t&quot;\n  input: &quot;generator/layer4/deconv2d/b/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/mul/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.6000000238418579\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;generator/layer4/mul/x&quot;\n  input: &quot;generator/layer4/deconv2d/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/Abs&quot;\n  op: &quot;Abs&quot;\n  input: &quot;generator/layer4/deconv2d/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/mul/1/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.4000000059604645\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/mul/1&quot;\n  op: &quot;Mul&quot;\n  input: &quot;generator/layer4/mul/1/x&quot;\n  input: &quot;generator/layer4/Abs&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/add&quot;\n  op: &quot;Add&quot;\n  input: &quot;generator/layer4/mul&quot;\n  input: &quot;generator/layer4/mul/1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\n';
              }
            </script>
            <link rel=&quot;import&quot; href=&quot;https://tensorboard.appspot.com/tf-graph-basic.build.html&quot; onload=load()>
            <div style=&quot;height:600px&quot;>
              <tf-graph-basic id=&quot;graph0.9779902055767046&quot;></tf-graph-basic>
            </div>
        "></iframe>
        


<a name="building-the-discriminator-for-the-generated-samples"></a>
## Building the Discriminator for the Generated Samples

Lastly, we need *another* discriminator which takes as input our generated images.  Recall the discriminator that we have made only takes as input our placeholder `X` which is for our actual training samples.  We'll use the same function for creating our discriminator and **reuse** the variables we already have.  This is the crucial part!  We aren't making *new* trainable variables, but reusing the ones we have.  We just create a new set of operations that takes as input our generated image.  So we'll have a whole new set of operations exactly like the ones we have created for our first discriminator.  But we are going to use the exact same variables as our first discriminator, so that we optimize the same values.


```python
D_fake = discriminator(G, reuse=True)
```

Now we can look at the graph and see the new discriminator inside the node for the discriminator.  You should see the original discriminator and a new graph of a discriminator within it, but all the weights are shared with the original discriminator.


```python
nb_utils.show_graph(graph.as_graph_def())
```



            <iframe seamless style="width:800px;height:620px;border:0" srcdoc="
            <script>
              function load() {
                document.getElementById(&quot;graph0.26567269497633117&quot;).pbtxt = 'node {\n  name: &quot;X&quot;\n  op: &quot;Placeholder&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: -1\n        }\n        dim {\n          size: 32\n        }\n        dim {\n          size: 32\n        }\n        dim {\n          size: 3\n        }\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/W/Initializer/random/uniform/shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        tensor_content: &quot;\\003\\000\\000\\000\\003\\000\\000\\000\\003\\000\\000\\000\\004\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/W/Initializer/random/uniform/min&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: -0.30860671401023865\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/W/Initializer/random/uniform/max&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.30860671401023865\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/W/Initializer/random/uniform/RandomUniform&quot;\n  op: &quot;RandomUniform&quot;\n  input: &quot;discriminator/layer1/0/W/Initializer/random/uniform/shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;seed&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;seed2&quot;\n    value {\n      i: 0\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/W/Initializer/random/uniform/sub&quot;\n  op: &quot;Sub&quot;\n  input: &quot;discriminator/layer1/0/W/Initializer/random/uniform/max&quot;\n  input: &quot;discriminator/layer1/0/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/W/Initializer/random/uniform/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer1/0/W/Initializer/random/uniform/RandomUniform&quot;\n  input: &quot;discriminator/layer1/0/W/Initializer/random/uniform/sub&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/W/Initializer/random/uniform&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/layer1/0/W/Initializer/random/uniform/mul&quot;\n  input: &quot;discriminator/layer1/0/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/W&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 3\n        }\n        dim {\n          size: 3\n        }\n        dim {\n          size: 3\n        }\n        dim {\n          size: 4\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/W/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;discriminator/layer1/0/W&quot;\n  input: &quot;discriminator/layer1/0/W/Initializer/random/uniform&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/W/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;discriminator/layer1/0/W&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;X&quot;\n  input: &quot;discriminator/layer1/0/W/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/b/Initializer/Const&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        float_val: 0.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/b&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 4\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/b/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;discriminator/layer1/0/b&quot;\n  input: &quot;discriminator/layer1/0/b/Initializer/Const&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/b/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;discriminator/layer1/0/b&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/b&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/h&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;discriminator/layer1/0/conv&quot;\n  input: &quot;discriminator/layer1/0/b/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/mul/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.6000000238418579\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer1/mul/x&quot;\n  input: &quot;discriminator/layer1/0/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/Abs&quot;\n  op: &quot;Abs&quot;\n  input: &quot;discriminator/layer1/0/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/mul/1/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.4000000059604645\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/mul/1&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer1/mul/1/x&quot;\n  input: &quot;discriminator/layer1/Abs&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/add&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/layer1/mul&quot;\n  input: &quot;discriminator/layer1/mul/1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/W/Initializer/random/uniform/shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        tensor_content: &quot;\\003\\000\\000\\000\\003\\000\\000\\000\\004\\000\\000\\000\\004\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/W/Initializer/random/uniform/min&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: -0.28867512941360474\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/W/Initializer/random/uniform/max&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.28867512941360474\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/W/Initializer/random/uniform/RandomUniform&quot;\n  op: &quot;RandomUniform&quot;\n  input: &quot;discriminator/layer2/1/W/Initializer/random/uniform/shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;seed&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;seed2&quot;\n    value {\n      i: 0\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/W/Initializer/random/uniform/sub&quot;\n  op: &quot;Sub&quot;\n  input: &quot;discriminator/layer2/1/W/Initializer/random/uniform/max&quot;\n  input: &quot;discriminator/layer2/1/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/W/Initializer/random/uniform/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer2/1/W/Initializer/random/uniform/RandomUniform&quot;\n  input: &quot;discriminator/layer2/1/W/Initializer/random/uniform/sub&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/W/Initializer/random/uniform&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/layer2/1/W/Initializer/random/uniform/mul&quot;\n  input: &quot;discriminator/layer2/1/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/W&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 3\n        }\n        dim {\n          size: 3\n        }\n        dim {\n          size: 4\n        }\n        dim {\n          size: 4\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/W/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;discriminator/layer2/1/W&quot;\n  input: &quot;discriminator/layer2/1/W/Initializer/random/uniform&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/W/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;discriminator/layer2/1/W&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;discriminator/layer1/add&quot;\n  input: &quot;discriminator/layer2/1/W/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/b/Initializer/Const&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        float_val: 0.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/b&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 4\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/b/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;discriminator/layer2/1/b&quot;\n  input: &quot;discriminator/layer2/1/b/Initializer/Const&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/b/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;discriminator/layer2/1/b&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/b&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/h&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;discriminator/layer2/1/conv&quot;\n  input: &quot;discriminator/layer2/1/b/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/mul/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.6000000238418579\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer2/mul/x&quot;\n  input: &quot;discriminator/layer2/1/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/Abs&quot;\n  op: &quot;Abs&quot;\n  input: &quot;discriminator/layer2/1/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/mul/1/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.4000000059604645\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/mul/1&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer2/mul/1/x&quot;\n  input: &quot;discriminator/layer2/Abs&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/add&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/layer2/mul&quot;\n  input: &quot;discriminator/layer2/mul/1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/W/Initializer/random/uniform/shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        tensor_content: &quot;\\003\\000\\000\\000\\003\\000\\000\\000\\004\\000\\000\\000\\004\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/W/Initializer/random/uniform/min&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: -0.28867512941360474\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/W/Initializer/random/uniform/max&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.28867512941360474\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/W/Initializer/random/uniform/RandomUniform&quot;\n  op: &quot;RandomUniform&quot;\n  input: &quot;discriminator/layer3/2/W/Initializer/random/uniform/shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;seed&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;seed2&quot;\n    value {\n      i: 0\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/W/Initializer/random/uniform/sub&quot;\n  op: &quot;Sub&quot;\n  input: &quot;discriminator/layer3/2/W/Initializer/random/uniform/max&quot;\n  input: &quot;discriminator/layer3/2/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/W/Initializer/random/uniform/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer3/2/W/Initializer/random/uniform/RandomUniform&quot;\n  input: &quot;discriminator/layer3/2/W/Initializer/random/uniform/sub&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/W/Initializer/random/uniform&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/layer3/2/W/Initializer/random/uniform/mul&quot;\n  input: &quot;discriminator/layer3/2/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/W&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 3\n        }\n        dim {\n          size: 3\n        }\n        dim {\n          size: 4\n        }\n        dim {\n          size: 4\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/W/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;discriminator/layer3/2/W&quot;\n  input: &quot;discriminator/layer3/2/W/Initializer/random/uniform&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/W/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;discriminator/layer3/2/W&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;discriminator/layer2/add&quot;\n  input: &quot;discriminator/layer3/2/W/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/b/Initializer/Const&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        float_val: 0.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/b&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 4\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/b/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;discriminator/layer3/2/b&quot;\n  input: &quot;discriminator/layer3/2/b/Initializer/Const&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/b/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;discriminator/layer3/2/b&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/b&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/h&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;discriminator/layer3/2/conv&quot;\n  input: &quot;discriminator/layer3/2/b/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/mul/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.6000000238418579\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer3/mul/x&quot;\n  input: &quot;discriminator/layer3/2/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/Abs&quot;\n  op: &quot;Abs&quot;\n  input: &quot;discriminator/layer3/2/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/mul/1/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.4000000059604645\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/mul/1&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer3/mul/1/x&quot;\n  input: &quot;discriminator/layer3/Abs&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/add&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/layer3/mul&quot;\n  input: &quot;discriminator/layer3/mul/1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/W/Initializer/random/uniform/shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        tensor_content: &quot;\\003\\000\\000\\000\\003\\000\\000\\000\\004\\000\\000\\000\\004\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/W/Initializer/random/uniform/min&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: -0.28867512941360474\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/W/Initializer/random/uniform/max&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.28867512941360474\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/W/Initializer/random/uniform/RandomUniform&quot;\n  op: &quot;RandomUniform&quot;\n  input: &quot;discriminator/layer4/3/W/Initializer/random/uniform/shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;seed&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;seed2&quot;\n    value {\n      i: 0\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/W/Initializer/random/uniform/sub&quot;\n  op: &quot;Sub&quot;\n  input: &quot;discriminator/layer4/3/W/Initializer/random/uniform/max&quot;\n  input: &quot;discriminator/layer4/3/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/W/Initializer/random/uniform/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer4/3/W/Initializer/random/uniform/RandomUniform&quot;\n  input: &quot;discriminator/layer4/3/W/Initializer/random/uniform/sub&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/W/Initializer/random/uniform&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/layer4/3/W/Initializer/random/uniform/mul&quot;\n  input: &quot;discriminator/layer4/3/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/W&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 3\n        }\n        dim {\n          size: 3\n        }\n        dim {\n          size: 4\n        }\n        dim {\n          size: 4\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/W/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;discriminator/layer4/3/W&quot;\n  input: &quot;discriminator/layer4/3/W/Initializer/random/uniform&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/W/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;discriminator/layer4/3/W&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;discriminator/layer3/add&quot;\n  input: &quot;discriminator/layer4/3/W/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/b/Initializer/Const&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        float_val: 0.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/b&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 4\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/b/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;discriminator/layer4/3/b&quot;\n  input: &quot;discriminator/layer4/3/b/Initializer/Const&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/b/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;discriminator/layer4/3/b&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/b&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/h&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;discriminator/layer4/3/conv&quot;\n  input: &quot;discriminator/layer4/3/b/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/mul/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.6000000238418579\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer4/mul/x&quot;\n  input: &quot;discriminator/layer4/3/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/Abs&quot;\n  op: &quot;Abs&quot;\n  input: &quot;discriminator/layer4/3/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/mul/1/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.4000000059604645\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/mul/1&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer4/mul/1/x&quot;\n  input: &quot;discriminator/layer4/Abs&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/add&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/layer4/mul&quot;\n  input: &quot;discriminator/layer4/mul/1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Reshape/shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 2\n          }\n        }\n        tensor_content: &quot;\\377\\377\\377\\377\\020\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Reshape&quot;\n  op: &quot;Reshape&quot;\n  input: &quot;discriminator/layer4/add&quot;\n  input: &quot;discriminator/Reshape/shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;Tshape&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/W/Initializer/random/uniform/shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 2\n          }\n        }\n        tensor_content: &quot;\\020\\000\\000\\000\\001\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/W/Initializer/random/uniform/min&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: -0.5940885543823242\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/W/Initializer/random/uniform/max&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.5940885543823242\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/W/Initializer/random/uniform/RandomUniform&quot;\n  op: &quot;RandomUniform&quot;\n  input: &quot;discriminator/Y/W/Initializer/random/uniform/shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;seed&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;seed2&quot;\n    value {\n      i: 0\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/W/Initializer/random/uniform/sub&quot;\n  op: &quot;Sub&quot;\n  input: &quot;discriminator/Y/W/Initializer/random/uniform/max&quot;\n  input: &quot;discriminator/Y/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/W/Initializer/random/uniform/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/Y/W/Initializer/random/uniform/RandomUniform&quot;\n  input: &quot;discriminator/Y/W/Initializer/random/uniform/sub&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/W/Initializer/random/uniform&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/Y/W/Initializer/random/uniform/mul&quot;\n  input: &quot;discriminator/Y/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/W&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 16\n        }\n        dim {\n          size: 1\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/W/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;discriminator/Y/W&quot;\n  input: &quot;discriminator/Y/W/Initializer/random/uniform&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/W/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;discriminator/Y/W&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/b/Initializer/Const&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1\n          }\n        }\n        float_val: 0.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/b&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 1\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/b/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;discriminator/Y/b&quot;\n  input: &quot;discriminator/Y/b/Initializer/Const&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/b/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;discriminator/Y/b&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/b&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/MatMul&quot;\n  op: &quot;MatMul&quot;\n  input: &quot;discriminator/Reshape&quot;\n  input: &quot;discriminator/Y/W/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;transpose_a&quot;\n    value {\n      b: false\n    }\n  }\n  attr {\n    key: &quot;transpose_b&quot;\n    value {\n      b: false\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/h&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;discriminator/Y/MatMul&quot;\n  input: &quot;discriminator/Y/b/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/Sigmoid&quot;\n  op: &quot;Sigmoid&quot;\n  input: &quot;discriminator/Y/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;Z&quot;\n  op: &quot;Placeholder&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: -1\n        }\n        dim {\n          size: 64\n        }\n      }\n    }\n  }\n}\nnode {\n  name: &quot;Reshape/shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        tensor_content: &quot;\\377\\377\\377\\377\\002\\000\\000\\000\\002\\000\\000\\000\\020\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;Reshape&quot;\n  op: &quot;Reshape&quot;\n  input: &quot;Z&quot;\n  input: &quot;Reshape/shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;Tshape&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/W/Initializer/random/uniform/shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer1/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        tensor_content: &quot;\\004\\000\\000\\000\\004\\000\\000\\0002\\000\\000\\000\\020\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/W/Initializer/random/uniform/min&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer1/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: -0.07537783682346344\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/W/Initializer/random/uniform/max&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer1/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.07537783682346344\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/W/Initializer/random/uniform/RandomUniform&quot;\n  op: &quot;RandomUniform&quot;\n  input: &quot;generator/layer1/deconv2d/W/Initializer/random/uniform/shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer1/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;seed&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;seed2&quot;\n    value {\n      i: 0\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/W/Initializer/random/uniform/sub&quot;\n  op: &quot;Sub&quot;\n  input: &quot;generator/layer1/deconv2d/W/Initializer/random/uniform/max&quot;\n  input: &quot;generator/layer1/deconv2d/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer1/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/W/Initializer/random/uniform/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;generator/layer1/deconv2d/W/Initializer/random/uniform/RandomUniform&quot;\n  input: &quot;generator/layer1/deconv2d/W/Initializer/random/uniform/sub&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer1/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/W/Initializer/random/uniform&quot;\n  op: &quot;Add&quot;\n  input: &quot;generator/layer1/deconv2d/W/Initializer/random/uniform/mul&quot;\n  input: &quot;generator/layer1/deconv2d/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer1/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/W&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer1/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 4\n        }\n        dim {\n          size: 4\n        }\n        dim {\n          size: 50\n        }\n        dim {\n          size: 16\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/W/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;generator/layer1/deconv2d/W&quot;\n  input: &quot;generator/layer1/deconv2d/W/Initializer/random/uniform&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer1/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/W/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;generator/layer1/deconv2d/W&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer1/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/Shape&quot;\n  op: &quot;Shape&quot;\n  input: &quot;Reshape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;out_type&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/strided/slice/stack&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 1\n          }\n        }\n        int_val: 0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/strided/slice/stack_1&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 1\n          }\n        }\n        int_val: 1\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/strided/slice/stack_2&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 1\n          }\n        }\n        int_val: 1\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/strided/slice&quot;\n  op: &quot;StridedSlice&quot;\n  input: &quot;generator/layer1/deconv2d/Shape&quot;\n  input: &quot;generator/layer1/deconv2d/strided/slice/stack&quot;\n  input: &quot;generator/layer1/deconv2d/strided/slice/stack_1&quot;\n  input: &quot;generator/layer1/deconv2d/strided/slice/stack_2&quot;\n  attr {\n    key: &quot;Index&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;begin_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;ellipsis_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;end_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;new_axis_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;shrink_axis_mask&quot;\n    value {\n      i: 1\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/stack/1&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 4\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/stack/2&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 4\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/stack/3&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 50\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/stack&quot;\n  op: &quot;Pack&quot;\n  input: &quot;generator/layer1/deconv2d/strided/slice&quot;\n  input: &quot;generator/layer1/deconv2d/stack/1&quot;\n  input: &quot;generator/layer1/deconv2d/stack/2&quot;\n  input: &quot;generator/layer1/deconv2d/stack/3&quot;\n  attr {\n    key: &quot;N&quot;\n    value {\n      i: 4\n    }\n  }\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;axis&quot;\n    value {\n      i: 0\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/conv/t&quot;\n  op: &quot;Conv2DBackpropInput&quot;\n  input: &quot;generator/layer1/deconv2d/stack&quot;\n  input: &quot;generator/layer1/deconv2d/W/read&quot;\n  input: &quot;Reshape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/b/Initializer/Const&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer1/deconv2d/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 50\n          }\n        }\n        float_val: 0.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/b&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer1/deconv2d/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 50\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/b/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;generator/layer1/deconv2d/b&quot;\n  input: &quot;generator/layer1/deconv2d/b/Initializer/Const&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer1/deconv2d/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/b/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;generator/layer1/deconv2d/b&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer1/deconv2d/b&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/h&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;generator/layer1/deconv2d/conv/t&quot;\n  input: &quot;generator/layer1/deconv2d/b/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/mul/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.6000000238418579\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;generator/layer1/mul/x&quot;\n  input: &quot;generator/layer1/deconv2d/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/Abs&quot;\n  op: &quot;Abs&quot;\n  input: &quot;generator/layer1/deconv2d/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/mul/1/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.4000000059604645\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/mul/1&quot;\n  op: &quot;Mul&quot;\n  input: &quot;generator/layer1/mul/1/x&quot;\n  input: &quot;generator/layer1/Abs&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/add&quot;\n  op: &quot;Add&quot;\n  input: &quot;generator/layer1/mul&quot;\n  input: &quot;generator/layer1/mul/1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/W/Initializer/random/uniform/shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer2/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        tensor_content: &quot;\\004\\000\\000\\000\\004\\000\\000\\0002\\000\\000\\0002\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/W/Initializer/random/uniform/min&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer2/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: -0.06123724207282066\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/W/Initializer/random/uniform/max&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer2/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.06123724207282066\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/W/Initializer/random/uniform/RandomUniform&quot;\n  op: &quot;RandomUniform&quot;\n  input: &quot;generator/layer2/deconv2d/W/Initializer/random/uniform/shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer2/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;seed&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;seed2&quot;\n    value {\n      i: 0\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/W/Initializer/random/uniform/sub&quot;\n  op: &quot;Sub&quot;\n  input: &quot;generator/layer2/deconv2d/W/Initializer/random/uniform/max&quot;\n  input: &quot;generator/layer2/deconv2d/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer2/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/W/Initializer/random/uniform/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;generator/layer2/deconv2d/W/Initializer/random/uniform/RandomUniform&quot;\n  input: &quot;generator/layer2/deconv2d/W/Initializer/random/uniform/sub&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer2/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/W/Initializer/random/uniform&quot;\n  op: &quot;Add&quot;\n  input: &quot;generator/layer2/deconv2d/W/Initializer/random/uniform/mul&quot;\n  input: &quot;generator/layer2/deconv2d/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer2/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/W&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer2/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 4\n        }\n        dim {\n          size: 4\n        }\n        dim {\n          size: 50\n        }\n        dim {\n          size: 50\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/W/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;generator/layer2/deconv2d/W&quot;\n  input: &quot;generator/layer2/deconv2d/W/Initializer/random/uniform&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer2/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/W/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;generator/layer2/deconv2d/W&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer2/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/Shape&quot;\n  op: &quot;Shape&quot;\n  input: &quot;generator/layer1/add&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;out_type&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/strided/slice/stack&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 1\n          }\n        }\n        int_val: 0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/strided/slice/stack_1&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 1\n          }\n        }\n        int_val: 1\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/strided/slice/stack_2&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 1\n          }\n        }\n        int_val: 1\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/strided/slice&quot;\n  op: &quot;StridedSlice&quot;\n  input: &quot;generator/layer2/deconv2d/Shape&quot;\n  input: &quot;generator/layer2/deconv2d/strided/slice/stack&quot;\n  input: &quot;generator/layer2/deconv2d/strided/slice/stack_1&quot;\n  input: &quot;generator/layer2/deconv2d/strided/slice/stack_2&quot;\n  attr {\n    key: &quot;Index&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;begin_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;ellipsis_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;end_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;new_axis_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;shrink_axis_mask&quot;\n    value {\n      i: 1\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/stack/1&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 8\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/stack/2&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 8\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/stack/3&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 50\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/stack&quot;\n  op: &quot;Pack&quot;\n  input: &quot;generator/layer2/deconv2d/strided/slice&quot;\n  input: &quot;generator/layer2/deconv2d/stack/1&quot;\n  input: &quot;generator/layer2/deconv2d/stack/2&quot;\n  input: &quot;generator/layer2/deconv2d/stack/3&quot;\n  attr {\n    key: &quot;N&quot;\n    value {\n      i: 4\n    }\n  }\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;axis&quot;\n    value {\n      i: 0\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/conv/t&quot;\n  op: &quot;Conv2DBackpropInput&quot;\n  input: &quot;generator/layer2/deconv2d/stack&quot;\n  input: &quot;generator/layer2/deconv2d/W/read&quot;\n  input: &quot;generator/layer1/add&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/b/Initializer/Const&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer2/deconv2d/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 50\n          }\n        }\n        float_val: 0.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/b&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer2/deconv2d/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 50\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/b/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;generator/layer2/deconv2d/b&quot;\n  input: &quot;generator/layer2/deconv2d/b/Initializer/Const&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer2/deconv2d/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/b/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;generator/layer2/deconv2d/b&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer2/deconv2d/b&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/h&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;generator/layer2/deconv2d/conv/t&quot;\n  input: &quot;generator/layer2/deconv2d/b/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/mul/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.6000000238418579\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;generator/layer2/mul/x&quot;\n  input: &quot;generator/layer2/deconv2d/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/Abs&quot;\n  op: &quot;Abs&quot;\n  input: &quot;generator/layer2/deconv2d/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/mul/1/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.4000000059604645\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/mul/1&quot;\n  op: &quot;Mul&quot;\n  input: &quot;generator/layer2/mul/1/x&quot;\n  input: &quot;generator/layer2/Abs&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/add&quot;\n  op: &quot;Add&quot;\n  input: &quot;generator/layer2/mul&quot;\n  input: &quot;generator/layer2/mul/1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/W/Initializer/random/uniform/shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer3/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        tensor_content: &quot;\\004\\000\\000\\000\\004\\000\\000\\0002\\000\\000\\0002\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/W/Initializer/random/uniform/min&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer3/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: -0.06123724207282066\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/W/Initializer/random/uniform/max&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer3/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.06123724207282066\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/W/Initializer/random/uniform/RandomUniform&quot;\n  op: &quot;RandomUniform&quot;\n  input: &quot;generator/layer3/deconv2d/W/Initializer/random/uniform/shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer3/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;seed&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;seed2&quot;\n    value {\n      i: 0\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/W/Initializer/random/uniform/sub&quot;\n  op: &quot;Sub&quot;\n  input: &quot;generator/layer3/deconv2d/W/Initializer/random/uniform/max&quot;\n  input: &quot;generator/layer3/deconv2d/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer3/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/W/Initializer/random/uniform/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;generator/layer3/deconv2d/W/Initializer/random/uniform/RandomUniform&quot;\n  input: &quot;generator/layer3/deconv2d/W/Initializer/random/uniform/sub&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer3/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/W/Initializer/random/uniform&quot;\n  op: &quot;Add&quot;\n  input: &quot;generator/layer3/deconv2d/W/Initializer/random/uniform/mul&quot;\n  input: &quot;generator/layer3/deconv2d/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer3/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/W&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer3/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 4\n        }\n        dim {\n          size: 4\n        }\n        dim {\n          size: 50\n        }\n        dim {\n          size: 50\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/W/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;generator/layer3/deconv2d/W&quot;\n  input: &quot;generator/layer3/deconv2d/W/Initializer/random/uniform&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer3/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/W/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;generator/layer3/deconv2d/W&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer3/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/Shape&quot;\n  op: &quot;Shape&quot;\n  input: &quot;generator/layer2/add&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;out_type&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/strided/slice/stack&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 1\n          }\n        }\n        int_val: 0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/strided/slice/stack_1&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 1\n          }\n        }\n        int_val: 1\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/strided/slice/stack_2&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 1\n          }\n        }\n        int_val: 1\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/strided/slice&quot;\n  op: &quot;StridedSlice&quot;\n  input: &quot;generator/layer3/deconv2d/Shape&quot;\n  input: &quot;generator/layer3/deconv2d/strided/slice/stack&quot;\n  input: &quot;generator/layer3/deconv2d/strided/slice/stack_1&quot;\n  input: &quot;generator/layer3/deconv2d/strided/slice/stack_2&quot;\n  attr {\n    key: &quot;Index&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;begin_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;ellipsis_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;end_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;new_axis_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;shrink_axis_mask&quot;\n    value {\n      i: 1\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/stack/1&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 16\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/stack/2&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 16\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/stack/3&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 50\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/stack&quot;\n  op: &quot;Pack&quot;\n  input: &quot;generator/layer3/deconv2d/strided/slice&quot;\n  input: &quot;generator/layer3/deconv2d/stack/1&quot;\n  input: &quot;generator/layer3/deconv2d/stack/2&quot;\n  input: &quot;generator/layer3/deconv2d/stack/3&quot;\n  attr {\n    key: &quot;N&quot;\n    value {\n      i: 4\n    }\n  }\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;axis&quot;\n    value {\n      i: 0\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/conv/t&quot;\n  op: &quot;Conv2DBackpropInput&quot;\n  input: &quot;generator/layer3/deconv2d/stack&quot;\n  input: &quot;generator/layer3/deconv2d/W/read&quot;\n  input: &quot;generator/layer2/add&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/b/Initializer/Const&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer3/deconv2d/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 50\n          }\n        }\n        float_val: 0.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/b&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer3/deconv2d/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 50\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/b/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;generator/layer3/deconv2d/b&quot;\n  input: &quot;generator/layer3/deconv2d/b/Initializer/Const&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer3/deconv2d/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/b/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;generator/layer3/deconv2d/b&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer3/deconv2d/b&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/h&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;generator/layer3/deconv2d/conv/t&quot;\n  input: &quot;generator/layer3/deconv2d/b/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/mul/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.6000000238418579\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;generator/layer3/mul/x&quot;\n  input: &quot;generator/layer3/deconv2d/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/Abs&quot;\n  op: &quot;Abs&quot;\n  input: &quot;generator/layer3/deconv2d/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/mul/1/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.4000000059604645\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/mul/1&quot;\n  op: &quot;Mul&quot;\n  input: &quot;generator/layer3/mul/1/x&quot;\n  input: &quot;generator/layer3/Abs&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/add&quot;\n  op: &quot;Add&quot;\n  input: &quot;generator/layer3/mul&quot;\n  input: &quot;generator/layer3/mul/1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/W/Initializer/random/uniform/shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer4/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        tensor_content: &quot;\\004\\000\\000\\000\\004\\000\\000\\000\\003\\000\\000\\0002\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/W/Initializer/random/uniform/min&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer4/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: -0.08411582559347153\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/W/Initializer/random/uniform/max&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer4/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.08411582559347153\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/W/Initializer/random/uniform/RandomUniform&quot;\n  op: &quot;RandomUniform&quot;\n  input: &quot;generator/layer4/deconv2d/W/Initializer/random/uniform/shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer4/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;seed&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;seed2&quot;\n    value {\n      i: 0\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/W/Initializer/random/uniform/sub&quot;\n  op: &quot;Sub&quot;\n  input: &quot;generator/layer4/deconv2d/W/Initializer/random/uniform/max&quot;\n  input: &quot;generator/layer4/deconv2d/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer4/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/W/Initializer/random/uniform/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;generator/layer4/deconv2d/W/Initializer/random/uniform/RandomUniform&quot;\n  input: &quot;generator/layer4/deconv2d/W/Initializer/random/uniform/sub&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer4/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/W/Initializer/random/uniform&quot;\n  op: &quot;Add&quot;\n  input: &quot;generator/layer4/deconv2d/W/Initializer/random/uniform/mul&quot;\n  input: &quot;generator/layer4/deconv2d/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer4/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/W&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer4/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 4\n        }\n        dim {\n          size: 4\n        }\n        dim {\n          size: 3\n        }\n        dim {\n          size: 50\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/W/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;generator/layer4/deconv2d/W&quot;\n  input: &quot;generator/layer4/deconv2d/W/Initializer/random/uniform&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer4/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/W/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;generator/layer4/deconv2d/W&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer4/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/Shape&quot;\n  op: &quot;Shape&quot;\n  input: &quot;generator/layer3/add&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;out_type&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/strided/slice/stack&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 1\n          }\n        }\n        int_val: 0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/strided/slice/stack_1&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 1\n          }\n        }\n        int_val: 1\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/strided/slice/stack_2&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 1\n          }\n        }\n        int_val: 1\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/strided/slice&quot;\n  op: &quot;StridedSlice&quot;\n  input: &quot;generator/layer4/deconv2d/Shape&quot;\n  input: &quot;generator/layer4/deconv2d/strided/slice/stack&quot;\n  input: &quot;generator/layer4/deconv2d/strided/slice/stack_1&quot;\n  input: &quot;generator/layer4/deconv2d/strided/slice/stack_2&quot;\n  attr {\n    key: &quot;Index&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;begin_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;ellipsis_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;end_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;new_axis_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;shrink_axis_mask&quot;\n    value {\n      i: 1\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/stack/1&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 32\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/stack/2&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 32\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/stack/3&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 3\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/stack&quot;\n  op: &quot;Pack&quot;\n  input: &quot;generator/layer4/deconv2d/strided/slice&quot;\n  input: &quot;generator/layer4/deconv2d/stack/1&quot;\n  input: &quot;generator/layer4/deconv2d/stack/2&quot;\n  input: &quot;generator/layer4/deconv2d/stack/3&quot;\n  attr {\n    key: &quot;N&quot;\n    value {\n      i: 4\n    }\n  }\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;axis&quot;\n    value {\n      i: 0\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/conv/t&quot;\n  op: &quot;Conv2DBackpropInput&quot;\n  input: &quot;generator/layer4/deconv2d/stack&quot;\n  input: &quot;generator/layer4/deconv2d/W/read&quot;\n  input: &quot;generator/layer3/add&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/b/Initializer/Const&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer4/deconv2d/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 3\n          }\n        }\n        float_val: 0.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/b&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer4/deconv2d/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 3\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/b/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;generator/layer4/deconv2d/b&quot;\n  input: &quot;generator/layer4/deconv2d/b/Initializer/Const&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer4/deconv2d/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/b/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;generator/layer4/deconv2d/b&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer4/deconv2d/b&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/h&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;generator/layer4/deconv2d/conv/t&quot;\n  input: &quot;generator/layer4/deconv2d/b/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/mul/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.6000000238418579\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;generator/layer4/mul/x&quot;\n  input: &quot;generator/layer4/deconv2d/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/Abs&quot;\n  op: &quot;Abs&quot;\n  input: &quot;generator/layer4/deconv2d/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/mul/1/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.4000000059604645\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/mul/1&quot;\n  op: &quot;Mul&quot;\n  input: &quot;generator/layer4/mul/1/x&quot;\n  input: &quot;generator/layer4/Abs&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/add&quot;\n  op: &quot;Add&quot;\n  input: &quot;generator/layer4/mul&quot;\n  input: &quot;generator/layer4/mul/1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer1/0/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;generator/layer4/add&quot;\n  input: &quot;discriminator/layer1/0/W/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer1/0/h&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;discriminator/1/layer1/0/conv&quot;\n  input: &quot;discriminator/layer1/0/b/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer1/mul/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.6000000238418579\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer1/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/1/layer1/mul/x&quot;\n  input: &quot;discriminator/1/layer1/0/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer1/Abs&quot;\n  op: &quot;Abs&quot;\n  input: &quot;discriminator/1/layer1/0/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer1/mul_1/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.4000000059604645\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer1/mul_1&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/1/layer1/mul_1/x&quot;\n  input: &quot;discriminator/1/layer1/Abs&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer1/add&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/1/layer1/mul&quot;\n  input: &quot;discriminator/1/layer1/mul_1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer2/1/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;discriminator/1/layer1/add&quot;\n  input: &quot;discriminator/layer2/1/W/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer2/1/h&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;discriminator/1/layer2/1/conv&quot;\n  input: &quot;discriminator/layer2/1/b/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer2/mul/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.6000000238418579\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer2/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/1/layer2/mul/x&quot;\n  input: &quot;discriminator/1/layer2/1/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer2/Abs&quot;\n  op: &quot;Abs&quot;\n  input: &quot;discriminator/1/layer2/1/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer2/mul_1/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.4000000059604645\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer2/mul_1&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/1/layer2/mul_1/x&quot;\n  input: &quot;discriminator/1/layer2/Abs&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer2/add&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/1/layer2/mul&quot;\n  input: &quot;discriminator/1/layer2/mul_1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer3/2/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;discriminator/1/layer2/add&quot;\n  input: &quot;discriminator/layer3/2/W/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer3/2/h&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;discriminator/1/layer3/2/conv&quot;\n  input: &quot;discriminator/layer3/2/b/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer3/mul/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.6000000238418579\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer3/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/1/layer3/mul/x&quot;\n  input: &quot;discriminator/1/layer3/2/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer3/Abs&quot;\n  op: &quot;Abs&quot;\n  input: &quot;discriminator/1/layer3/2/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer3/mul_1/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.4000000059604645\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer3/mul_1&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/1/layer3/mul_1/x&quot;\n  input: &quot;discriminator/1/layer3/Abs&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer3/add&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/1/layer3/mul&quot;\n  input: &quot;discriminator/1/layer3/mul_1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer4/3/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;discriminator/1/layer3/add&quot;\n  input: &quot;discriminator/layer4/3/W/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer4/3/h&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;discriminator/1/layer4/3/conv&quot;\n  input: &quot;discriminator/layer4/3/b/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer4/mul/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.6000000238418579\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer4/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/1/layer4/mul/x&quot;\n  input: &quot;discriminator/1/layer4/3/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer4/Abs&quot;\n  op: &quot;Abs&quot;\n  input: &quot;discriminator/1/layer4/3/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer4/mul_1/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.4000000059604645\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer4/mul_1&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/1/layer4/mul_1/x&quot;\n  input: &quot;discriminator/1/layer4/Abs&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer4/add&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/1/layer4/mul&quot;\n  input: &quot;discriminator/1/layer4/mul_1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/Reshape/shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 2\n          }\n        }\n        tensor_content: &quot;\\377\\377\\377\\377\\020\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/Reshape&quot;\n  op: &quot;Reshape&quot;\n  input: &quot;discriminator/1/layer4/add&quot;\n  input: &quot;discriminator/1/Reshape/shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;Tshape&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/Y/MatMul&quot;\n  op: &quot;MatMul&quot;\n  input: &quot;discriminator/1/Reshape&quot;\n  input: &quot;discriminator/Y/W/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;transpose_a&quot;\n    value {\n      b: false\n    }\n  }\n  attr {\n    key: &quot;transpose_b&quot;\n    value {\n      b: false\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/Y/h&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;discriminator/1/Y/MatMul&quot;\n  input: &quot;discriminator/Y/b/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/Y/Sigmoid&quot;\n  op: &quot;Sigmoid&quot;\n  input: &quot;discriminator/1/Y/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\n';
              }
            </script>
            <link rel=&quot;import&quot; href=&quot;https://tensorboard.appspot.com/tf-graph-basic.build.html&quot; onload=load()>
            <div style=&quot;height:600px&quot;>
              <tf-graph-basic id=&quot;graph0.26567269497633117&quot;></tf-graph-basic>
            </div>
        "></iframe>
        


<a name="gan-loss-functions"></a>
## GAN Loss Functions

We now have all the components to our network.  We just have to train it.  This is the notoriously tricky bit.  We will have 3 different loss measures instead of our typical network with just a single loss.  We'll later connect each of these loss measures to two optimizers, one for the generator and another for the discriminator, and then pin them against each other and see which one wins!  Exciting times!

Recall from Session 3's Supervised Network, we created a binary classification task: music or speech.  We again have a binary classification task: real or fake.  So our loss metric will again use the binary cross entropy to measure the loss of our three different modules: the generator, the discriminator for our real images, and the discriminator for our generated images.

To find out the loss function for our generator network, answer the question, what makes the generator successful?  Successfully fooling the discriminator.  When does that happen?  When the discriminator for the fake samples produces all ones.  So our binary cross entropy measure will measure the cross entropy with our predicted distribution and the true distribution which has all ones.


```python
with tf.variable_scope('loss/generator'):
    loss_G = tf.reduce_mean(utils.binary_cross_entropy(D_fake, tf.ones_like(D_fake)))
```

What we've just written is a loss function for our generator.  The generator is optimized when the discriminator for the generated samples produces all ones.  In contrast to the generator, the discriminator will have 2 measures to optimize.  One which is the opposite of what we have just written above, as well as 1 more measure for the real samples.  Try writing these two losses and we'll combine them using their average.  We want to optimize the Discriminator for the real samples producing all 1s, and the Discriminator for the fake samples producing all 0s:

<h3><font color='red'>TODO! COMPLETE THIS SECTION!</font></h3>


```python
with tf.variable_scope('loss/discriminator/real'):
    loss_D_real = utils.binary_cross_entropy(
            D_real, tf.ones_like(D_real), name='loss_D_real')
with tf.variable_scope('loss/discriminator/fake'):
    loss_D_fake = utils.binary_cross_entropy(
            D_fake, tf.zeros_like(D_fake), name='loss_D_fake')
with tf.variable_scope('loss/discriminator'):
    loss_D = utils.tf.reduce_mean((loss_D_real + loss_D_fake) / 2)
```


```python
nb_utils.show_graph(graph.as_graph_def())
```



            <iframe seamless style="width:800px;height:620px;border:0" srcdoc="
            <script>
              function load() {
                document.getElementById(&quot;graph0.06732885234738473&quot;).pbtxt = 'node {\n  name: &quot;X&quot;\n  op: &quot;Placeholder&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: -1\n        }\n        dim {\n          size: 32\n        }\n        dim {\n          size: 32\n        }\n        dim {\n          size: 3\n        }\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/W/Initializer/random/uniform/shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        tensor_content: &quot;\\003\\000\\000\\000\\003\\000\\000\\000\\003\\000\\000\\000\\004\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/W/Initializer/random/uniform/min&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: -0.30860671401023865\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/W/Initializer/random/uniform/max&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.30860671401023865\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/W/Initializer/random/uniform/RandomUniform&quot;\n  op: &quot;RandomUniform&quot;\n  input: &quot;discriminator/layer1/0/W/Initializer/random/uniform/shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;seed&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;seed2&quot;\n    value {\n      i: 0\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/W/Initializer/random/uniform/sub&quot;\n  op: &quot;Sub&quot;\n  input: &quot;discriminator/layer1/0/W/Initializer/random/uniform/max&quot;\n  input: &quot;discriminator/layer1/0/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/W/Initializer/random/uniform/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer1/0/W/Initializer/random/uniform/RandomUniform&quot;\n  input: &quot;discriminator/layer1/0/W/Initializer/random/uniform/sub&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/W/Initializer/random/uniform&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/layer1/0/W/Initializer/random/uniform/mul&quot;\n  input: &quot;discriminator/layer1/0/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/W&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 3\n        }\n        dim {\n          size: 3\n        }\n        dim {\n          size: 3\n        }\n        dim {\n          size: 4\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/W/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;discriminator/layer1/0/W&quot;\n  input: &quot;discriminator/layer1/0/W/Initializer/random/uniform&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/W/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;discriminator/layer1/0/W&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;X&quot;\n  input: &quot;discriminator/layer1/0/W/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/b/Initializer/Const&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        float_val: 0.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/b&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 4\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/b/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;discriminator/layer1/0/b&quot;\n  input: &quot;discriminator/layer1/0/b/Initializer/Const&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/b/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;discriminator/layer1/0/b&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer1/0/b&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/0/h&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;discriminator/layer1/0/conv&quot;\n  input: &quot;discriminator/layer1/0/b/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/mul/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.6000000238418579\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer1/mul/x&quot;\n  input: &quot;discriminator/layer1/0/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/Abs&quot;\n  op: &quot;Abs&quot;\n  input: &quot;discriminator/layer1/0/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/mul/1/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.4000000059604645\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/mul/1&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer1/mul/1/x&quot;\n  input: &quot;discriminator/layer1/Abs&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer1/add&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/layer1/mul&quot;\n  input: &quot;discriminator/layer1/mul/1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/W/Initializer/random/uniform/shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        tensor_content: &quot;\\003\\000\\000\\000\\003\\000\\000\\000\\004\\000\\000\\000\\004\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/W/Initializer/random/uniform/min&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: -0.28867512941360474\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/W/Initializer/random/uniform/max&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.28867512941360474\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/W/Initializer/random/uniform/RandomUniform&quot;\n  op: &quot;RandomUniform&quot;\n  input: &quot;discriminator/layer2/1/W/Initializer/random/uniform/shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;seed&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;seed2&quot;\n    value {\n      i: 0\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/W/Initializer/random/uniform/sub&quot;\n  op: &quot;Sub&quot;\n  input: &quot;discriminator/layer2/1/W/Initializer/random/uniform/max&quot;\n  input: &quot;discriminator/layer2/1/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/W/Initializer/random/uniform/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer2/1/W/Initializer/random/uniform/RandomUniform&quot;\n  input: &quot;discriminator/layer2/1/W/Initializer/random/uniform/sub&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/W/Initializer/random/uniform&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/layer2/1/W/Initializer/random/uniform/mul&quot;\n  input: &quot;discriminator/layer2/1/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/W&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 3\n        }\n        dim {\n          size: 3\n        }\n        dim {\n          size: 4\n        }\n        dim {\n          size: 4\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/W/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;discriminator/layer2/1/W&quot;\n  input: &quot;discriminator/layer2/1/W/Initializer/random/uniform&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/W/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;discriminator/layer2/1/W&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;discriminator/layer1/add&quot;\n  input: &quot;discriminator/layer2/1/W/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/b/Initializer/Const&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        float_val: 0.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/b&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 4\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/b/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;discriminator/layer2/1/b&quot;\n  input: &quot;discriminator/layer2/1/b/Initializer/Const&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/b/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;discriminator/layer2/1/b&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer2/1/b&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/1/h&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;discriminator/layer2/1/conv&quot;\n  input: &quot;discriminator/layer2/1/b/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/mul/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.6000000238418579\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer2/mul/x&quot;\n  input: &quot;discriminator/layer2/1/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/Abs&quot;\n  op: &quot;Abs&quot;\n  input: &quot;discriminator/layer2/1/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/mul/1/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.4000000059604645\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/mul/1&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer2/mul/1/x&quot;\n  input: &quot;discriminator/layer2/Abs&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer2/add&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/layer2/mul&quot;\n  input: &quot;discriminator/layer2/mul/1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/W/Initializer/random/uniform/shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        tensor_content: &quot;\\003\\000\\000\\000\\003\\000\\000\\000\\004\\000\\000\\000\\004\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/W/Initializer/random/uniform/min&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: -0.28867512941360474\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/W/Initializer/random/uniform/max&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.28867512941360474\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/W/Initializer/random/uniform/RandomUniform&quot;\n  op: &quot;RandomUniform&quot;\n  input: &quot;discriminator/layer3/2/W/Initializer/random/uniform/shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;seed&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;seed2&quot;\n    value {\n      i: 0\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/W/Initializer/random/uniform/sub&quot;\n  op: &quot;Sub&quot;\n  input: &quot;discriminator/layer3/2/W/Initializer/random/uniform/max&quot;\n  input: &quot;discriminator/layer3/2/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/W/Initializer/random/uniform/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer3/2/W/Initializer/random/uniform/RandomUniform&quot;\n  input: &quot;discriminator/layer3/2/W/Initializer/random/uniform/sub&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/W/Initializer/random/uniform&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/layer3/2/W/Initializer/random/uniform/mul&quot;\n  input: &quot;discriminator/layer3/2/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/W&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 3\n        }\n        dim {\n          size: 3\n        }\n        dim {\n          size: 4\n        }\n        dim {\n          size: 4\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/W/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;discriminator/layer3/2/W&quot;\n  input: &quot;discriminator/layer3/2/W/Initializer/random/uniform&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/W/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;discriminator/layer3/2/W&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;discriminator/layer2/add&quot;\n  input: &quot;discriminator/layer3/2/W/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/b/Initializer/Const&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        float_val: 0.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/b&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 4\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/b/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;discriminator/layer3/2/b&quot;\n  input: &quot;discriminator/layer3/2/b/Initializer/Const&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/b/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;discriminator/layer3/2/b&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer3/2/b&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/2/h&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;discriminator/layer3/2/conv&quot;\n  input: &quot;discriminator/layer3/2/b/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/mul/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.6000000238418579\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer3/mul/x&quot;\n  input: &quot;discriminator/layer3/2/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/Abs&quot;\n  op: &quot;Abs&quot;\n  input: &quot;discriminator/layer3/2/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/mul/1/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.4000000059604645\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/mul/1&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer3/mul/1/x&quot;\n  input: &quot;discriminator/layer3/Abs&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer3/add&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/layer3/mul&quot;\n  input: &quot;discriminator/layer3/mul/1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/W/Initializer/random/uniform/shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        tensor_content: &quot;\\003\\000\\000\\000\\003\\000\\000\\000\\004\\000\\000\\000\\004\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/W/Initializer/random/uniform/min&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: -0.28867512941360474\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/W/Initializer/random/uniform/max&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.28867512941360474\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/W/Initializer/random/uniform/RandomUniform&quot;\n  op: &quot;RandomUniform&quot;\n  input: &quot;discriminator/layer4/3/W/Initializer/random/uniform/shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;seed&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;seed2&quot;\n    value {\n      i: 0\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/W/Initializer/random/uniform/sub&quot;\n  op: &quot;Sub&quot;\n  input: &quot;discriminator/layer4/3/W/Initializer/random/uniform/max&quot;\n  input: &quot;discriminator/layer4/3/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/W/Initializer/random/uniform/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer4/3/W/Initializer/random/uniform/RandomUniform&quot;\n  input: &quot;discriminator/layer4/3/W/Initializer/random/uniform/sub&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/W/Initializer/random/uniform&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/layer4/3/W/Initializer/random/uniform/mul&quot;\n  input: &quot;discriminator/layer4/3/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/W&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 3\n        }\n        dim {\n          size: 3\n        }\n        dim {\n          size: 4\n        }\n        dim {\n          size: 4\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/W/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;discriminator/layer4/3/W&quot;\n  input: &quot;discriminator/layer4/3/W/Initializer/random/uniform&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/W/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;discriminator/layer4/3/W&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;discriminator/layer3/add&quot;\n  input: &quot;discriminator/layer4/3/W/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/b/Initializer/Const&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        float_val: 0.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/b&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 4\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/b/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;discriminator/layer4/3/b&quot;\n  input: &quot;discriminator/layer4/3/b/Initializer/Const&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/b/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;discriminator/layer4/3/b&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/layer4/3/b&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/3/h&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;discriminator/layer4/3/conv&quot;\n  input: &quot;discriminator/layer4/3/b/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/mul/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.6000000238418579\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer4/mul/x&quot;\n  input: &quot;discriminator/layer4/3/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/Abs&quot;\n  op: &quot;Abs&quot;\n  input: &quot;discriminator/layer4/3/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/mul/1/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.4000000059604645\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/mul/1&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/layer4/mul/1/x&quot;\n  input: &quot;discriminator/layer4/Abs&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/layer4/add&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/layer4/mul&quot;\n  input: &quot;discriminator/layer4/mul/1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Reshape/shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 2\n          }\n        }\n        tensor_content: &quot;\\377\\377\\377\\377\\020\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Reshape&quot;\n  op: &quot;Reshape&quot;\n  input: &quot;discriminator/layer4/add&quot;\n  input: &quot;discriminator/Reshape/shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;Tshape&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/W/Initializer/random/uniform/shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 2\n          }\n        }\n        tensor_content: &quot;\\020\\000\\000\\000\\001\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/W/Initializer/random/uniform/min&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: -0.5940885543823242\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/W/Initializer/random/uniform/max&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.5940885543823242\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/W/Initializer/random/uniform/RandomUniform&quot;\n  op: &quot;RandomUniform&quot;\n  input: &quot;discriminator/Y/W/Initializer/random/uniform/shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;seed&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;seed2&quot;\n    value {\n      i: 0\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/W/Initializer/random/uniform/sub&quot;\n  op: &quot;Sub&quot;\n  input: &quot;discriminator/Y/W/Initializer/random/uniform/max&quot;\n  input: &quot;discriminator/Y/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/W/Initializer/random/uniform/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/Y/W/Initializer/random/uniform/RandomUniform&quot;\n  input: &quot;discriminator/Y/W/Initializer/random/uniform/sub&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/W/Initializer/random/uniform&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/Y/W/Initializer/random/uniform/mul&quot;\n  input: &quot;discriminator/Y/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/W&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 16\n        }\n        dim {\n          size: 1\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/W/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;discriminator/Y/W&quot;\n  input: &quot;discriminator/Y/W/Initializer/random/uniform&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/W/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;discriminator/Y/W&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/b/Initializer/Const&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1\n          }\n        }\n        float_val: 0.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/b&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 1\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/b/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;discriminator/Y/b&quot;\n  input: &quot;discriminator/Y/b/Initializer/Const&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/b/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;discriminator/Y/b&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@discriminator/Y/b&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/MatMul&quot;\n  op: &quot;MatMul&quot;\n  input: &quot;discriminator/Reshape&quot;\n  input: &quot;discriminator/Y/W/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;transpose_a&quot;\n    value {\n      b: false\n    }\n  }\n  attr {\n    key: &quot;transpose_b&quot;\n    value {\n      b: false\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/h&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;discriminator/Y/MatMul&quot;\n  input: &quot;discriminator/Y/b/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/Y/Sigmoid&quot;\n  op: &quot;Sigmoid&quot;\n  input: &quot;discriminator/Y/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;Z&quot;\n  op: &quot;Placeholder&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: -1\n        }\n        dim {\n          size: 64\n        }\n      }\n    }\n  }\n}\nnode {\n  name: &quot;Reshape/shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        tensor_content: &quot;\\377\\377\\377\\377\\002\\000\\000\\000\\002\\000\\000\\000\\020\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;Reshape&quot;\n  op: &quot;Reshape&quot;\n  input: &quot;Z&quot;\n  input: &quot;Reshape/shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;Tshape&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/W/Initializer/random/uniform/shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer1/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        tensor_content: &quot;\\004\\000\\000\\000\\004\\000\\000\\0002\\000\\000\\000\\020\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/W/Initializer/random/uniform/min&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer1/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: -0.07537783682346344\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/W/Initializer/random/uniform/max&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer1/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.07537783682346344\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/W/Initializer/random/uniform/RandomUniform&quot;\n  op: &quot;RandomUniform&quot;\n  input: &quot;generator/layer1/deconv2d/W/Initializer/random/uniform/shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer1/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;seed&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;seed2&quot;\n    value {\n      i: 0\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/W/Initializer/random/uniform/sub&quot;\n  op: &quot;Sub&quot;\n  input: &quot;generator/layer1/deconv2d/W/Initializer/random/uniform/max&quot;\n  input: &quot;generator/layer1/deconv2d/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer1/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/W/Initializer/random/uniform/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;generator/layer1/deconv2d/W/Initializer/random/uniform/RandomUniform&quot;\n  input: &quot;generator/layer1/deconv2d/W/Initializer/random/uniform/sub&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer1/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/W/Initializer/random/uniform&quot;\n  op: &quot;Add&quot;\n  input: &quot;generator/layer1/deconv2d/W/Initializer/random/uniform/mul&quot;\n  input: &quot;generator/layer1/deconv2d/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer1/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/W&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer1/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 4\n        }\n        dim {\n          size: 4\n        }\n        dim {\n          size: 50\n        }\n        dim {\n          size: 16\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/W/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;generator/layer1/deconv2d/W&quot;\n  input: &quot;generator/layer1/deconv2d/W/Initializer/random/uniform&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer1/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/W/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;generator/layer1/deconv2d/W&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer1/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/Shape&quot;\n  op: &quot;Shape&quot;\n  input: &quot;Reshape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;out_type&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/strided/slice/stack&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 1\n          }\n        }\n        int_val: 0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/strided/slice/stack_1&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 1\n          }\n        }\n        int_val: 1\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/strided/slice/stack_2&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 1\n          }\n        }\n        int_val: 1\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/strided/slice&quot;\n  op: &quot;StridedSlice&quot;\n  input: &quot;generator/layer1/deconv2d/Shape&quot;\n  input: &quot;generator/layer1/deconv2d/strided/slice/stack&quot;\n  input: &quot;generator/layer1/deconv2d/strided/slice/stack_1&quot;\n  input: &quot;generator/layer1/deconv2d/strided/slice/stack_2&quot;\n  attr {\n    key: &quot;Index&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;begin_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;ellipsis_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;end_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;new_axis_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;shrink_axis_mask&quot;\n    value {\n      i: 1\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/stack/1&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 4\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/stack/2&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 4\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/stack/3&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 50\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/stack&quot;\n  op: &quot;Pack&quot;\n  input: &quot;generator/layer1/deconv2d/strided/slice&quot;\n  input: &quot;generator/layer1/deconv2d/stack/1&quot;\n  input: &quot;generator/layer1/deconv2d/stack/2&quot;\n  input: &quot;generator/layer1/deconv2d/stack/3&quot;\n  attr {\n    key: &quot;N&quot;\n    value {\n      i: 4\n    }\n  }\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;axis&quot;\n    value {\n      i: 0\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/conv/t&quot;\n  op: &quot;Conv2DBackpropInput&quot;\n  input: &quot;generator/layer1/deconv2d/stack&quot;\n  input: &quot;generator/layer1/deconv2d/W/read&quot;\n  input: &quot;Reshape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/b/Initializer/Const&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer1/deconv2d/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 50\n          }\n        }\n        float_val: 0.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/b&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer1/deconv2d/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 50\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/b/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;generator/layer1/deconv2d/b&quot;\n  input: &quot;generator/layer1/deconv2d/b/Initializer/Const&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer1/deconv2d/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/b/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;generator/layer1/deconv2d/b&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer1/deconv2d/b&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/deconv2d/h&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;generator/layer1/deconv2d/conv/t&quot;\n  input: &quot;generator/layer1/deconv2d/b/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/mul/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.6000000238418579\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;generator/layer1/mul/x&quot;\n  input: &quot;generator/layer1/deconv2d/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/Abs&quot;\n  op: &quot;Abs&quot;\n  input: &quot;generator/layer1/deconv2d/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/mul/1/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.4000000059604645\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/mul/1&quot;\n  op: &quot;Mul&quot;\n  input: &quot;generator/layer1/mul/1/x&quot;\n  input: &quot;generator/layer1/Abs&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer1/add&quot;\n  op: &quot;Add&quot;\n  input: &quot;generator/layer1/mul&quot;\n  input: &quot;generator/layer1/mul/1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/W/Initializer/random/uniform/shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer2/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        tensor_content: &quot;\\004\\000\\000\\000\\004\\000\\000\\0002\\000\\000\\0002\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/W/Initializer/random/uniform/min&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer2/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: -0.06123724207282066\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/W/Initializer/random/uniform/max&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer2/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.06123724207282066\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/W/Initializer/random/uniform/RandomUniform&quot;\n  op: &quot;RandomUniform&quot;\n  input: &quot;generator/layer2/deconv2d/W/Initializer/random/uniform/shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer2/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;seed&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;seed2&quot;\n    value {\n      i: 0\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/W/Initializer/random/uniform/sub&quot;\n  op: &quot;Sub&quot;\n  input: &quot;generator/layer2/deconv2d/W/Initializer/random/uniform/max&quot;\n  input: &quot;generator/layer2/deconv2d/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer2/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/W/Initializer/random/uniform/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;generator/layer2/deconv2d/W/Initializer/random/uniform/RandomUniform&quot;\n  input: &quot;generator/layer2/deconv2d/W/Initializer/random/uniform/sub&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer2/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/W/Initializer/random/uniform&quot;\n  op: &quot;Add&quot;\n  input: &quot;generator/layer2/deconv2d/W/Initializer/random/uniform/mul&quot;\n  input: &quot;generator/layer2/deconv2d/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer2/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/W&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer2/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 4\n        }\n        dim {\n          size: 4\n        }\n        dim {\n          size: 50\n        }\n        dim {\n          size: 50\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/W/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;generator/layer2/deconv2d/W&quot;\n  input: &quot;generator/layer2/deconv2d/W/Initializer/random/uniform&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer2/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/W/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;generator/layer2/deconv2d/W&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer2/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/Shape&quot;\n  op: &quot;Shape&quot;\n  input: &quot;generator/layer1/add&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;out_type&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/strided/slice/stack&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 1\n          }\n        }\n        int_val: 0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/strided/slice/stack_1&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 1\n          }\n        }\n        int_val: 1\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/strided/slice/stack_2&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 1\n          }\n        }\n        int_val: 1\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/strided/slice&quot;\n  op: &quot;StridedSlice&quot;\n  input: &quot;generator/layer2/deconv2d/Shape&quot;\n  input: &quot;generator/layer2/deconv2d/strided/slice/stack&quot;\n  input: &quot;generator/layer2/deconv2d/strided/slice/stack_1&quot;\n  input: &quot;generator/layer2/deconv2d/strided/slice/stack_2&quot;\n  attr {\n    key: &quot;Index&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;begin_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;ellipsis_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;end_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;new_axis_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;shrink_axis_mask&quot;\n    value {\n      i: 1\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/stack/1&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 8\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/stack/2&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 8\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/stack/3&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 50\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/stack&quot;\n  op: &quot;Pack&quot;\n  input: &quot;generator/layer2/deconv2d/strided/slice&quot;\n  input: &quot;generator/layer2/deconv2d/stack/1&quot;\n  input: &quot;generator/layer2/deconv2d/stack/2&quot;\n  input: &quot;generator/layer2/deconv2d/stack/3&quot;\n  attr {\n    key: &quot;N&quot;\n    value {\n      i: 4\n    }\n  }\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;axis&quot;\n    value {\n      i: 0\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/conv/t&quot;\n  op: &quot;Conv2DBackpropInput&quot;\n  input: &quot;generator/layer2/deconv2d/stack&quot;\n  input: &quot;generator/layer2/deconv2d/W/read&quot;\n  input: &quot;generator/layer1/add&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/b/Initializer/Const&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer2/deconv2d/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 50\n          }\n        }\n        float_val: 0.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/b&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer2/deconv2d/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 50\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/b/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;generator/layer2/deconv2d/b&quot;\n  input: &quot;generator/layer2/deconv2d/b/Initializer/Const&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer2/deconv2d/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/b/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;generator/layer2/deconv2d/b&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer2/deconv2d/b&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/deconv2d/h&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;generator/layer2/deconv2d/conv/t&quot;\n  input: &quot;generator/layer2/deconv2d/b/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/mul/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.6000000238418579\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;generator/layer2/mul/x&quot;\n  input: &quot;generator/layer2/deconv2d/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/Abs&quot;\n  op: &quot;Abs&quot;\n  input: &quot;generator/layer2/deconv2d/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/mul/1/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.4000000059604645\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/mul/1&quot;\n  op: &quot;Mul&quot;\n  input: &quot;generator/layer2/mul/1/x&quot;\n  input: &quot;generator/layer2/Abs&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer2/add&quot;\n  op: &quot;Add&quot;\n  input: &quot;generator/layer2/mul&quot;\n  input: &quot;generator/layer2/mul/1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/W/Initializer/random/uniform/shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer3/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        tensor_content: &quot;\\004\\000\\000\\000\\004\\000\\000\\0002\\000\\000\\0002\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/W/Initializer/random/uniform/min&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer3/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: -0.06123724207282066\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/W/Initializer/random/uniform/max&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer3/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.06123724207282066\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/W/Initializer/random/uniform/RandomUniform&quot;\n  op: &quot;RandomUniform&quot;\n  input: &quot;generator/layer3/deconv2d/W/Initializer/random/uniform/shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer3/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;seed&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;seed2&quot;\n    value {\n      i: 0\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/W/Initializer/random/uniform/sub&quot;\n  op: &quot;Sub&quot;\n  input: &quot;generator/layer3/deconv2d/W/Initializer/random/uniform/max&quot;\n  input: &quot;generator/layer3/deconv2d/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer3/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/W/Initializer/random/uniform/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;generator/layer3/deconv2d/W/Initializer/random/uniform/RandomUniform&quot;\n  input: &quot;generator/layer3/deconv2d/W/Initializer/random/uniform/sub&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer3/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/W/Initializer/random/uniform&quot;\n  op: &quot;Add&quot;\n  input: &quot;generator/layer3/deconv2d/W/Initializer/random/uniform/mul&quot;\n  input: &quot;generator/layer3/deconv2d/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer3/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/W&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer3/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 4\n        }\n        dim {\n          size: 4\n        }\n        dim {\n          size: 50\n        }\n        dim {\n          size: 50\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/W/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;generator/layer3/deconv2d/W&quot;\n  input: &quot;generator/layer3/deconv2d/W/Initializer/random/uniform&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer3/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/W/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;generator/layer3/deconv2d/W&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer3/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/Shape&quot;\n  op: &quot;Shape&quot;\n  input: &quot;generator/layer2/add&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;out_type&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/strided/slice/stack&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 1\n          }\n        }\n        int_val: 0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/strided/slice/stack_1&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 1\n          }\n        }\n        int_val: 1\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/strided/slice/stack_2&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 1\n          }\n        }\n        int_val: 1\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/strided/slice&quot;\n  op: &quot;StridedSlice&quot;\n  input: &quot;generator/layer3/deconv2d/Shape&quot;\n  input: &quot;generator/layer3/deconv2d/strided/slice/stack&quot;\n  input: &quot;generator/layer3/deconv2d/strided/slice/stack_1&quot;\n  input: &quot;generator/layer3/deconv2d/strided/slice/stack_2&quot;\n  attr {\n    key: &quot;Index&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;begin_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;ellipsis_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;end_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;new_axis_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;shrink_axis_mask&quot;\n    value {\n      i: 1\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/stack/1&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 16\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/stack/2&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 16\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/stack/3&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 50\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/stack&quot;\n  op: &quot;Pack&quot;\n  input: &quot;generator/layer3/deconv2d/strided/slice&quot;\n  input: &quot;generator/layer3/deconv2d/stack/1&quot;\n  input: &quot;generator/layer3/deconv2d/stack/2&quot;\n  input: &quot;generator/layer3/deconv2d/stack/3&quot;\n  attr {\n    key: &quot;N&quot;\n    value {\n      i: 4\n    }\n  }\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;axis&quot;\n    value {\n      i: 0\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/conv/t&quot;\n  op: &quot;Conv2DBackpropInput&quot;\n  input: &quot;generator/layer3/deconv2d/stack&quot;\n  input: &quot;generator/layer3/deconv2d/W/read&quot;\n  input: &quot;generator/layer2/add&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/b/Initializer/Const&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer3/deconv2d/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 50\n          }\n        }\n        float_val: 0.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/b&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer3/deconv2d/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 50\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/b/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;generator/layer3/deconv2d/b&quot;\n  input: &quot;generator/layer3/deconv2d/b/Initializer/Const&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer3/deconv2d/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/b/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;generator/layer3/deconv2d/b&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer3/deconv2d/b&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/deconv2d/h&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;generator/layer3/deconv2d/conv/t&quot;\n  input: &quot;generator/layer3/deconv2d/b/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/mul/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.6000000238418579\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;generator/layer3/mul/x&quot;\n  input: &quot;generator/layer3/deconv2d/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/Abs&quot;\n  op: &quot;Abs&quot;\n  input: &quot;generator/layer3/deconv2d/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/mul/1/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.4000000059604645\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/mul/1&quot;\n  op: &quot;Mul&quot;\n  input: &quot;generator/layer3/mul/1/x&quot;\n  input: &quot;generator/layer3/Abs&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer3/add&quot;\n  op: &quot;Add&quot;\n  input: &quot;generator/layer3/mul&quot;\n  input: &quot;generator/layer3/mul/1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/W/Initializer/random/uniform/shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer4/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 4\n          }\n        }\n        tensor_content: &quot;\\004\\000\\000\\000\\004\\000\\000\\000\\003\\000\\000\\0002\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/W/Initializer/random/uniform/min&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer4/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: -0.08411582559347153\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/W/Initializer/random/uniform/max&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer4/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.08411582559347153\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/W/Initializer/random/uniform/RandomUniform&quot;\n  op: &quot;RandomUniform&quot;\n  input: &quot;generator/layer4/deconv2d/W/Initializer/random/uniform/shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer4/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;seed&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;seed2&quot;\n    value {\n      i: 0\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/W/Initializer/random/uniform/sub&quot;\n  op: &quot;Sub&quot;\n  input: &quot;generator/layer4/deconv2d/W/Initializer/random/uniform/max&quot;\n  input: &quot;generator/layer4/deconv2d/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer4/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/W/Initializer/random/uniform/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;generator/layer4/deconv2d/W/Initializer/random/uniform/RandomUniform&quot;\n  input: &quot;generator/layer4/deconv2d/W/Initializer/random/uniform/sub&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer4/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/W/Initializer/random/uniform&quot;\n  op: &quot;Add&quot;\n  input: &quot;generator/layer4/deconv2d/W/Initializer/random/uniform/mul&quot;\n  input: &quot;generator/layer4/deconv2d/W/Initializer/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer4/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/W&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer4/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 4\n        }\n        dim {\n          size: 4\n        }\n        dim {\n          size: 3\n        }\n        dim {\n          size: 50\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/W/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;generator/layer4/deconv2d/W&quot;\n  input: &quot;generator/layer4/deconv2d/W/Initializer/random/uniform&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer4/deconv2d/W&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/W/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;generator/layer4/deconv2d/W&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer4/deconv2d/W&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/Shape&quot;\n  op: &quot;Shape&quot;\n  input: &quot;generator/layer3/add&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;out_type&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/strided/slice/stack&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 1\n          }\n        }\n        int_val: 0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/strided/slice/stack_1&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 1\n          }\n        }\n        int_val: 1\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/strided/slice/stack_2&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 1\n          }\n        }\n        int_val: 1\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/strided/slice&quot;\n  op: &quot;StridedSlice&quot;\n  input: &quot;generator/layer4/deconv2d/Shape&quot;\n  input: &quot;generator/layer4/deconv2d/strided/slice/stack&quot;\n  input: &quot;generator/layer4/deconv2d/strided/slice/stack_1&quot;\n  input: &quot;generator/layer4/deconv2d/strided/slice/stack_2&quot;\n  attr {\n    key: &quot;Index&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;begin_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;ellipsis_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;end_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;new_axis_mask&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;shrink_axis_mask&quot;\n    value {\n      i: 1\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/stack/1&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 32\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/stack/2&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 32\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/stack/3&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 3\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/stack&quot;\n  op: &quot;Pack&quot;\n  input: &quot;generator/layer4/deconv2d/strided/slice&quot;\n  input: &quot;generator/layer4/deconv2d/stack/1&quot;\n  input: &quot;generator/layer4/deconv2d/stack/2&quot;\n  input: &quot;generator/layer4/deconv2d/stack/3&quot;\n  attr {\n    key: &quot;N&quot;\n    value {\n      i: 4\n    }\n  }\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;axis&quot;\n    value {\n      i: 0\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/conv/t&quot;\n  op: &quot;Conv2DBackpropInput&quot;\n  input: &quot;generator/layer4/deconv2d/stack&quot;\n  input: &quot;generator/layer4/deconv2d/W/read&quot;\n  input: &quot;generator/layer3/add&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/b/Initializer/Const&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer4/deconv2d/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 3\n          }\n        }\n        float_val: 0.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/b&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer4/deconv2d/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 3\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/b/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;generator/layer4/deconv2d/b&quot;\n  input: &quot;generator/layer4/deconv2d/b/Initializer/Const&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer4/deconv2d/b&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/b/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;generator/layer4/deconv2d/b&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@generator/layer4/deconv2d/b&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/deconv2d/h&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;generator/layer4/deconv2d/conv/t&quot;\n  input: &quot;generator/layer4/deconv2d/b/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/mul/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.6000000238418579\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;generator/layer4/mul/x&quot;\n  input: &quot;generator/layer4/deconv2d/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/Abs&quot;\n  op: &quot;Abs&quot;\n  input: &quot;generator/layer4/deconv2d/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/mul/1/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.4000000059604645\n      }\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/mul/1&quot;\n  op: &quot;Mul&quot;\n  input: &quot;generator/layer4/mul/1/x&quot;\n  input: &quot;generator/layer4/Abs&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;generator/layer4/add&quot;\n  op: &quot;Add&quot;\n  input: &quot;generator/layer4/mul&quot;\n  input: &quot;generator/layer4/mul/1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer1/0/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;generator/layer4/add&quot;\n  input: &quot;discriminator/layer1/0/W/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer1/0/h&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;discriminator/1/layer1/0/conv&quot;\n  input: &quot;discriminator/layer1/0/b/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer1/mul/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.6000000238418579\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer1/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/1/layer1/mul/x&quot;\n  input: &quot;discriminator/1/layer1/0/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer1/Abs&quot;\n  op: &quot;Abs&quot;\n  input: &quot;discriminator/1/layer1/0/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer1/mul_1/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.4000000059604645\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer1/mul_1&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/1/layer1/mul_1/x&quot;\n  input: &quot;discriminator/1/layer1/Abs&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer1/add&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/1/layer1/mul&quot;\n  input: &quot;discriminator/1/layer1/mul_1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer2/1/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;discriminator/1/layer1/add&quot;\n  input: &quot;discriminator/layer2/1/W/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer2/1/h&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;discriminator/1/layer2/1/conv&quot;\n  input: &quot;discriminator/layer2/1/b/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer2/mul/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.6000000238418579\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer2/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/1/layer2/mul/x&quot;\n  input: &quot;discriminator/1/layer2/1/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer2/Abs&quot;\n  op: &quot;Abs&quot;\n  input: &quot;discriminator/1/layer2/1/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer2/mul_1/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.4000000059604645\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer2/mul_1&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/1/layer2/mul_1/x&quot;\n  input: &quot;discriminator/1/layer2/Abs&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer2/add&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/1/layer2/mul&quot;\n  input: &quot;discriminator/1/layer2/mul_1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer3/2/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;discriminator/1/layer2/add&quot;\n  input: &quot;discriminator/layer3/2/W/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer3/2/h&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;discriminator/1/layer3/2/conv&quot;\n  input: &quot;discriminator/layer3/2/b/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer3/mul/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.6000000238418579\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer3/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/1/layer3/mul/x&quot;\n  input: &quot;discriminator/1/layer3/2/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer3/Abs&quot;\n  op: &quot;Abs&quot;\n  input: &quot;discriminator/1/layer3/2/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer3/mul_1/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.4000000059604645\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer3/mul_1&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/1/layer3/mul_1/x&quot;\n  input: &quot;discriminator/1/layer3/Abs&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer3/add&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/1/layer3/mul&quot;\n  input: &quot;discriminator/1/layer3/mul_1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer4/3/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;discriminator/1/layer3/add&quot;\n  input: &quot;discriminator/layer4/3/W/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer4/3/h&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;discriminator/1/layer4/3/conv&quot;\n  input: &quot;discriminator/layer4/3/b/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer4/mul/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.6000000238418579\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer4/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/1/layer4/mul/x&quot;\n  input: &quot;discriminator/1/layer4/3/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer4/Abs&quot;\n  op: &quot;Abs&quot;\n  input: &quot;discriminator/1/layer4/3/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer4/mul_1/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.4000000059604645\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer4/mul_1&quot;\n  op: &quot;Mul&quot;\n  input: &quot;discriminator/1/layer4/mul_1/x&quot;\n  input: &quot;discriminator/1/layer4/Abs&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/layer4/add&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/1/layer4/mul&quot;\n  input: &quot;discriminator/1/layer4/mul_1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/Reshape/shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 2\n          }\n        }\n        tensor_content: &quot;\\377\\377\\377\\377\\020\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/Reshape&quot;\n  op: &quot;Reshape&quot;\n  input: &quot;discriminator/1/layer4/add&quot;\n  input: &quot;discriminator/1/Reshape/shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;Tshape&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/Y/MatMul&quot;\n  op: &quot;MatMul&quot;\n  input: &quot;discriminator/1/Reshape&quot;\n  input: &quot;discriminator/Y/W/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;transpose_a&quot;\n    value {\n      b: false\n    }\n  }\n  attr {\n    key: &quot;transpose_b&quot;\n    value {\n      b: false\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/Y/h&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;discriminator/1/Y/MatMul&quot;\n  input: &quot;discriminator/Y/b/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;discriminator/1/Y/Sigmoid&quot;\n  op: &quot;Sigmoid&quot;\n  input: &quot;discriminator/1/Y/h&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;loss/generator/ones/like/Shape&quot;\n  op: &quot;Shape&quot;\n  input: &quot;discriminator/1/Y/Sigmoid&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;out_type&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n}\nnode {\n  name: &quot;loss/generator/ones/like/Const&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 1.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;loss/generator/ones/like&quot;\n  op: &quot;Fill&quot;\n  input: &quot;loss/generator/ones/like/Shape&quot;\n  input: &quot;loss/generator/ones/like/Const&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;loss/generator/bce/add/y&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 9.999999960041972e-13\n      }\n    }\n  }\n}\nnode {\n  name: &quot;loss/generator/bce/add&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/1/Y/Sigmoid&quot;\n  input: &quot;loss/generator/bce/add/y&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;loss/generator/bce/Log&quot;\n  op: &quot;Log&quot;\n  input: &quot;loss/generator/bce/add&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;loss/generator/bce/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;loss/generator/ones/like&quot;\n  input: &quot;loss/generator/bce/Log&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;loss/generator/bce/sub/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 1.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;loss/generator/bce/sub&quot;\n  op: &quot;Sub&quot;\n  input: &quot;loss/generator/bce/sub/x&quot;\n  input: &quot;loss/generator/ones/like&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;loss/generator/bce/sub/1/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 1.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;loss/generator/bce/sub/1&quot;\n  op: &quot;Sub&quot;\n  input: &quot;loss/generator/bce/sub/1/x&quot;\n  input: &quot;discriminator/1/Y/Sigmoid&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;loss/generator/bce/add/1/y&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 9.999999960041972e-13\n      }\n    }\n  }\n}\nnode {\n  name: &quot;loss/generator/bce/add/1&quot;\n  op: &quot;Add&quot;\n  input: &quot;loss/generator/bce/sub/1&quot;\n  input: &quot;loss/generator/bce/add/1/y&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;loss/generator/bce/Log/1&quot;\n  op: &quot;Log&quot;\n  input: &quot;loss/generator/bce/add/1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;loss/generator/bce/mul/1&quot;\n  op: &quot;Mul&quot;\n  input: &quot;loss/generator/bce/sub&quot;\n  input: &quot;loss/generator/bce/Log/1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;loss/generator/bce/add/2&quot;\n  op: &quot;Add&quot;\n  input: &quot;loss/generator/bce/mul&quot;\n  input: &quot;loss/generator/bce/mul/1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;loss/generator/bce/Neg&quot;\n  op: &quot;Neg&quot;\n  input: &quot;loss/generator/bce/add/2&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;loss/generator/Const&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 2\n          }\n        }\n        tensor_content: &quot;\\000\\000\\000\\000\\001\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;loss/generator/Mean&quot;\n  op: &quot;Mean&quot;\n  input: &quot;loss/generator/bce/Neg&quot;\n  input: &quot;loss/generator/Const&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;Tidx&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;keep_dims&quot;\n    value {\n      b: false\n    }\n  }\n}\nnode {\n  name: &quot;loss/discriminator/real/ones/like/Shape&quot;\n  op: &quot;Shape&quot;\n  input: &quot;discriminator/Y/Sigmoid&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;out_type&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n}\nnode {\n  name: &quot;loss/discriminator/real/ones/like/Const&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 1.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;loss/discriminator/real/ones/like&quot;\n  op: &quot;Fill&quot;\n  input: &quot;loss/discriminator/real/ones/like/Shape&quot;\n  input: &quot;loss/discriminator/real/ones/like/Const&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;loss/discriminator/real/loss/D_real/add/y&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 9.999999960041972e-13\n      }\n    }\n  }\n}\nnode {\n  name: &quot;loss/discriminator/real/loss/D_real/add&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/Y/Sigmoid&quot;\n  input: &quot;loss/discriminator/real/loss/D_real/add/y&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;loss/discriminator/real/loss/D_real/Log&quot;\n  op: &quot;Log&quot;\n  input: &quot;loss/discriminator/real/loss/D_real/add&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;loss/discriminator/real/loss/D_real/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;loss/discriminator/real/ones/like&quot;\n  input: &quot;loss/discriminator/real/loss/D_real/Log&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;loss/discriminator/real/loss/D_real/sub/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 1.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;loss/discriminator/real/loss/D_real/sub&quot;\n  op: &quot;Sub&quot;\n  input: &quot;loss/discriminator/real/loss/D_real/sub/x&quot;\n  input: &quot;loss/discriminator/real/ones/like&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;loss/discriminator/real/loss/D_real/sub_1/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 1.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;loss/discriminator/real/loss/D_real/sub_1&quot;\n  op: &quot;Sub&quot;\n  input: &quot;loss/discriminator/real/loss/D_real/sub_1/x&quot;\n  input: &quot;discriminator/Y/Sigmoid&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;loss/discriminator/real/loss/D_real/add_1/y&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 9.999999960041972e-13\n      }\n    }\n  }\n}\nnode {\n  name: &quot;loss/discriminator/real/loss/D_real/add_1&quot;\n  op: &quot;Add&quot;\n  input: &quot;loss/discriminator/real/loss/D_real/sub_1&quot;\n  input: &quot;loss/discriminator/real/loss/D_real/add_1/y&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;loss/discriminator/real/loss/D_real/Log_1&quot;\n  op: &quot;Log&quot;\n  input: &quot;loss/discriminator/real/loss/D_real/add_1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;loss/discriminator/real/loss/D_real/mul_1&quot;\n  op: &quot;Mul&quot;\n  input: &quot;loss/discriminator/real/loss/D_real/sub&quot;\n  input: &quot;loss/discriminator/real/loss/D_real/Log_1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;loss/discriminator/real/loss/D_real/add_2&quot;\n  op: &quot;Add&quot;\n  input: &quot;loss/discriminator/real/loss/D_real/mul&quot;\n  input: &quot;loss/discriminator/real/loss/D_real/mul_1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;loss/discriminator/real/loss/D_real/Neg&quot;\n  op: &quot;Neg&quot;\n  input: &quot;loss/discriminator/real/loss/D_real/add_2&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;loss/discriminator/fake/zeros/like&quot;\n  op: &quot;ZerosLike&quot;\n  input: &quot;discriminator/1/Y/Sigmoid&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;loss/discriminator/fake/loss/D_fake/add/y&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 9.999999960041972e-13\n      }\n    }\n  }\n}\nnode {\n  name: &quot;loss/discriminator/fake/loss/D_fake/add&quot;\n  op: &quot;Add&quot;\n  input: &quot;discriminator/1/Y/Sigmoid&quot;\n  input: &quot;loss/discriminator/fake/loss/D_fake/add/y&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;loss/discriminator/fake/loss/D_fake/Log&quot;\n  op: &quot;Log&quot;\n  input: &quot;loss/discriminator/fake/loss/D_fake/add&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;loss/discriminator/fake/loss/D_fake/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;loss/discriminator/fake/zeros/like&quot;\n  input: &quot;loss/discriminator/fake/loss/D_fake/Log&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;loss/discriminator/fake/loss/D_fake/sub/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 1.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;loss/discriminator/fake/loss/D_fake/sub&quot;\n  op: &quot;Sub&quot;\n  input: &quot;loss/discriminator/fake/loss/D_fake/sub/x&quot;\n  input: &quot;loss/discriminator/fake/zeros/like&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;loss/discriminator/fake/loss/D_fake/sub_1/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 1.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;loss/discriminator/fake/loss/D_fake/sub_1&quot;\n  op: &quot;Sub&quot;\n  input: &quot;loss/discriminator/fake/loss/D_fake/sub_1/x&quot;\n  input: &quot;discriminator/1/Y/Sigmoid&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;loss/discriminator/fake/loss/D_fake/add_1/y&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 9.999999960041972e-13\n      }\n    }\n  }\n}\nnode {\n  name: &quot;loss/discriminator/fake/loss/D_fake/add_1&quot;\n  op: &quot;Add&quot;\n  input: &quot;loss/discriminator/fake/loss/D_fake/sub_1&quot;\n  input: &quot;loss/discriminator/fake/loss/D_fake/add_1/y&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;loss/discriminator/fake/loss/D_fake/Log_1&quot;\n  op: &quot;Log&quot;\n  input: &quot;loss/discriminator/fake/loss/D_fake/add_1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;loss/discriminator/fake/loss/D_fake/mul_1&quot;\n  op: &quot;Mul&quot;\n  input: &quot;loss/discriminator/fake/loss/D_fake/sub&quot;\n  input: &quot;loss/discriminator/fake/loss/D_fake/Log_1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;loss/discriminator/fake/loss/D_fake/add_2&quot;\n  op: &quot;Add&quot;\n  input: &quot;loss/discriminator/fake/loss/D_fake/mul&quot;\n  input: &quot;loss/discriminator/fake/loss/D_fake/mul_1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;loss/discriminator/fake/loss/D_fake/Neg&quot;\n  op: &quot;Neg&quot;\n  input: &quot;loss/discriminator/fake/loss/D_fake/add_2&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;loss/discriminator/add&quot;\n  op: &quot;Add&quot;\n  input: &quot;loss/discriminator/real/loss/D_real/Neg&quot;\n  input: &quot;loss/discriminator/fake/loss/D_fake/Neg&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;loss/discriminator/truediv/y&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 2.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;loss/discriminator/truediv&quot;\n  op: &quot;RealDiv&quot;\n  input: &quot;loss/discriminator/add&quot;\n  input: &quot;loss/discriminator/truediv/y&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;loss/discriminator/Const&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 2\n          }\n        }\n        tensor_content: &quot;\\000\\000\\000\\000\\001\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;loss/discriminator/Mean&quot;\n  op: &quot;Mean&quot;\n  input: &quot;loss/discriminator/truediv&quot;\n  input: &quot;loss/discriminator/Const&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;Tidx&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;keep_dims&quot;\n    value {\n      b: false\n    }\n  }\n}\n';
              }
            </script>
            <link rel=&quot;import&quot; href=&quot;https://tensorboard.appspot.com/tf-graph-basic.build.html&quot; onload=load()>
            <div style=&quot;height:600px&quot;>
              <tf-graph-basic id=&quot;graph0.06732885234738473&quot;></tf-graph-basic>
            </div>
        "></iframe>
        


With our loss functions, we can create an optimizer for the discriminator and generator:

<a name="building-the-optimizers-w-regularization"></a>
## Building the Optimizers w/ Regularization

We're almost ready to create our optimizers.  We just need to do one extra thing.  Recall that our loss for our generator has a flow from the generator through the discriminator.  If we are training both the generator and the discriminator, we have two measures which both try to optimize the discriminator, but in opposite ways: the generator's loss would try to optimize the discriminator to be bad at its job, and the discriminator's loss would try to optimize it to be good at its job.  This would be counter-productive, trying to optimize opposing losses.  What we want is for the generator to get better, and the discriminator to get better.  Not for the discriminator to get better, then get worse, then get better, etc... The way we do this is when we optimize our generator, we let the gradient flow through the discriminator, but we do not update the variables in the discriminator.  Let's try and grab just the discriminator variables and just the generator variables below:

<h3><font color='red'>TODO! COMPLETE THIS SECTION!</font></h3>


```python
# Grab just the variables corresponding to the discriminator
# and just the generator:
vars_d = [v for v in tf.trainable_variables()
          if v.name.startswith('discriminator')]
print('Training discriminator variables:')
[print(v.name) for v in tf.trainable_variables()
 if v.name.startswith('discriminator')]

vars_g = [v for v in tf.trainable_variables()
          if v.name.startswith('generator')]
print('Training generator variables:')
[print(v.name) for v in tf.trainable_variables()
 if v.name.startswith('generator')]
```

    Training discriminator variables:
    discriminator/layer1/0/W:0
    discriminator/layer1/0/b:0
    discriminator/layer2/1/W:0
    discriminator/layer2/1/b:0
    discriminator/layer3/2/W:0
    discriminator/layer3/2/b:0
    discriminator/layer4/3/W:0
    discriminator/layer4/3/b:0
    discriminator/Y/W:0
    discriminator/Y/b:0
    Training generator variables:
    generator/layer1/deconv2d/W:0
    generator/layer1/deconv2d/b:0
    generator/layer2/deconv2d/W:0
    generator/layer2/deconv2d/b:0
    generator/layer3/deconv2d/W:0
    generator/layer3/deconv2d/b:0
    generator/layer4/deconv2d/W:0
    generator/layer4/deconv2d/b:0
    




    [None, None, None, None, None, None, None, None]



We can also apply regularization to our network.  This will penalize weights in the network for growing too large.


```python
d_reg = tf.contrib.layers.apply_regularization(
    tf.contrib.layers.l2_regularizer(1e-6), vars_d)
g_reg = tf.contrib.layers.apply_regularization(
    tf.contrib.layers.l2_regularizer(1e-6), vars_g)
```

The last thing you may want to try is creating a separate learning rate for each of your generator and discriminator optimizers like so:


```python
learning_rate = 0.0001

lr_g = tf.placeholder(tf.float32, shape=[], name='learning_rate_g')
lr_d = tf.placeholder(tf.float32, shape=[], name='learning_rate_d')
```

Now you can feed the placeholders to your optimizers.  If you run into errors creating these, then you likely have a problem with your graph's definition!  Be sure to go back and reset the default graph and check the sizes of your different operations/placeholders.

With your optimizers, you can now train the network by "running" the optimizer variables with your session.  You'll need to set the `var_list` parameter of the `minimize` function to only train the variables for the discriminator and same for the generator's optimizer:

<h3><font color='red'>TODO! COMPLETE THIS SECTION!</font></h3>


```python
opt_g = tf.train.AdamOptimizer(learning_rate=lr_g).minimize(loss_G + g_reg, var_list=vars_g)
```


```python
opt_d = tf.train.AdamOptimizer(learning_rate=lr_d).minimize(loss_D + d_reg, var_list=vars_d)
```

<a name="loading-a-dataset"></a>
## Loading a Dataset

Let's use the Celeb Dataset just for demonstration purposes.  In Part 2, you can explore using your own dataset.  This code is exactly the same as we did in Session 3's homework with the VAE.

<h3><font color='red'>TODO! COMPLETE THIS SECTION!</font></h3>


```python
# You'll want to change this to your own data if you end up training your own GAN.
batch_size = 64
n_epochs = 10
crop_shape = [n_pixels, n_pixels, 3]
crop_factor = 0.8
input_shape = [218, 178, 3]

files = datasets.CELEB()
batch = dataset_utils.create_input_pipeline(
        files=files,
        batch_size=batch_size,
        n_epochs=n_epochs,
        crop_shape=crop_shape,
        crop_factor=crop_factor,
        shape=input_shape)
```

    It does not look like you have downloaded the entire Celeb Dataset.
    Try downloading the dataset from the "Aligned and Cropped" link located here (imgs/img_align_celeba.zip [1.34 GB]): http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
    

<a name="training"></a>
## Training

We'll now go through the setup of training the network. We won't actually spend the time to train the network but just see how it would be done.  This is because in Part 2, we'll see an extension to this network which makes it much easier to train.


```python
ckpt_name = './gan.ckpt'

sess = tf.Session()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator()
tf.get_default_graph().finalize()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

if os.path.exists(ckpt_name + '.index') or os.path.exists(ckpt_name):
    saver.restore(sess, ckpt_name)
    print("VAE model restored.")
```


```python
n_examples = 10

zs = np.random.uniform(0.0, 1.0, [4, n_latent]).astype(np.float32)
zs = utils.make_latent_manifold(zs, n_examples)
```

<a name="equilibrium"></a>
## Equilibrium

 Equilibrium is at 0.693.  Why?  Consider what the cost is measuring, the binary cross entropy.  If we have random guesses, then we have as many 0s as we have 1s.  And on average, we'll be 50% correct.  The binary cross entropy is:

\begin{align}
\sum_i \text{X}_i * \text{log}(\tilde{\text{X}}_i) + (1 - \text{X}_i) * \text{log}(1 - \tilde{\text{X}}_i)
\end{align}

Which is written out in tensorflow as:

```python
(-(x * tf.log(z) + (1. - x) * tf.log(1. - z)))
```

Where `x` is the discriminator's prediction of the true distribution, in the case of GANs, the input images, and `z` is the discriminator's prediction of the generated images corresponding to the mathematical notation of $\tilde{\text{X}}$.  We sum over all features, but in the case of the discriminator, we have just 1 feature, the guess of whether it is a true image or not.  If our discriminator guesses at chance, i.e. 0.5,  then we'd have something like:

\begin{align}
0.5 * \text{log}(0.5) + (1 - 0.5) * \text{log}(1 - 0.5) = -0.693
\end{align}

So this is what we'd expect at the start of learning and from a game theoretic point of view, where we want things to remain.  So unlike our previous networks, where our loss continues to drop closer and closer to 0, we want our loss to waver around this value as much as possible, and hope for the best.


```python
equilibrium = 0.693
margin = 0.2
```

When we go to train the network, we switch back and forth between each optimizer, feeding in the appropriate values for each optimizer.  The `opt_g` optimizer only requires the `Z` and `lr_g` placeholders, while the `opt_d` optimizer requires the `X`, `Z`, and `lr_d` placeholders.

Don't train this network for very long because GANs are a huge pain to train and require a lot of fiddling.  They very easily get stuck in their adversarial process, or get overtaken by one or the other, resulting in a useless model.  What you need to develop is a steady equilibrium that optimizes both.  That will likely take two weeks just trying to get the GAN to train and not have enough time for the rest of the assignment.  They require a lot of memory/cpu and can take many days to train once you have settled on an architecture/training process/dataset.  Just let it run for a short time and then interrupt the kernel (don't restart!), then continue to the next cell.

From there, we'll go over an extension to the GAN which uses a VAE like we used in Session 3.  By using this extra network, we can actually train a better model in a fraction of the time and with much more ease!  But the network's definition is a bit more complicated.  Let's see how the GAN is trained first and then we'll train the VAE/GAN network instead.  While training, the "real" and "fake" cost will be printed out.  See how this cost wavers around the equilibrium and how we enforce it to try and stay around there by including a margin and some simple logic for updates.  This is highly experimental and the research does not have a good answer for the best practice on how to train a GAN.  I.e., some people will set the learning rate to some ratio of the performance between fake/real networks, others will have a fixed update schedule but train the generator twice and the discriminator only once.


```python
t_i = 0
batch_i = 0
epoch_i = 0
n_files = len(files)

if not os.path.exists('imgs'):
    os.makedirs('imgs')
    
while epoch_i < n_epochs:

    batch_i += 1
    batch_xs = sess.run(batch) / 255.0
    batch_zs = np.random.uniform(
        0.0, 1.0, [batch_size, n_latent]).astype(np.float32)

    real_cost, fake_cost = sess.run([
        loss_D_real, loss_D_fake],
        feed_dict={
            X: batch_xs,
            Z: batch_zs})
    real_cost = np.mean(real_cost)
    fake_cost = np.mean(fake_cost)
    
    if (batch_i % 20) == 0:
        print(batch_i, 'real:', real_cost, '/ fake:', fake_cost)

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
        sess.run(opt_g,
            feed_dict={
                Z: batch_zs,
                lr_g: learning_rate})
    if dis_update:
        sess.run(opt_d,
            feed_dict={
                X: batch_xs,
                Z: batch_zs,
                lr_d: learning_rate})

    if batch_i % (n_files // batch_size) == 0:
        batch_i = 0
        epoch_i += 1
        print('---------- EPOCH:', epoch_i)
        
        # Plot example reconstructions from latent layer
        recon = sess.run(G, feed_dict={Z: zs})

        recon = np.clip(recon, 0, 1)
        m1 = utils.montage(recon.reshape([-1] + crop_shape),
                'imgs/manifold_%08d.png' % t_i)

        recon = sess.run(G, feed_dict={Z: batch_zs})

        recon = np.clip(recon, 0, 1)
        m2 = utils.montage(recon.reshape([-1] + crop_shape),
                'imgs/reconstructions_%08d.png' % t_i)
        
        fig, axs = plt.subplots(1, 2, figsize=(15, 10))
        axs[0].imshow(m1)
        axs[1].imshow(m2)
        plt.show()
        t_i += 1

        # Save the variables to disk.
        save_path = saver.save(sess, "./" + ckpt_name,
                               global_step=batch_i,
                               write_meta_graph=False)
        print("Model saved in file: %s" % save_path)
```

    20 real: 0.610018 / fake: 0.701282
    40 real: 0.59012 / fake: 0.735685
    60 real: 0.57531 / fake: 0.872558
    80 real: 0.575185 / fake: 0.904108
    100 real: 0.576498 / fake: 0.895758
    120 real: 0.569521 / fake: 0.896676
    140 real: 0.586066 / fake: 0.899881
    160 real: 0.595045 / fake: 0.896333
    180 real: 0.594869 / fake: 0.894851
    200 real: 0.615708 / fake: 0.899969
    220 real: 0.626863 / fake: 0.894958
    240 real: 0.645443 / fake: 0.908076
    260 real: 0.679017 / fake: 0.901956
    280 real: 0.705914 / fake: 0.889628
    300 real: 0.713932 / fake: 0.911184
    320 real: 0.741027 / fake: 0.891837
    340 real: 0.77093 / fake: 0.903226
    360 real: 0.765869 / fake: 0.891092
    380 real: 0.755707 / fake: 0.892907
    400 real: 0.747859 / fake: 0.885563
    420 real: 0.749773 / fake: 0.884459
    440 real: 0.742776 / fake: 0.891763
    460 real: 0.745482 / fake: 0.887041
    480 real: 0.735987 / fake: 0.898228
    500 real: 0.73978 / fake: 0.89963
    520 real: 0.738648 / fake: 0.894251
    540 real: 0.73707 / fake: 0.891182
    560 real: 0.733811 / fake: 0.901089
    580 real: 0.736428 / fake: 0.90044
    600 real: 0.736163 / fake: 0.893182
    620 real: 0.734336 / fake: 0.892952
    640 real: 0.733653 / fake: 0.89136
    660 real: 0.732559 / fake: 0.893785
    680 real: 0.733041 / fake: 0.896349
    700 real: 0.733426 / fake: 0.896167
    720 real: 0.731545 / fake: 0.89471
    740 real: 0.732247 / fake: 0.89616
    760 real: 0.731728 / fake: 0.892505
    780 real: 0.73297 / fake: 0.895345
    800 real: 0.733238 / fake: 0.887661
    820 real: 0.733912 / fake: 0.889461
    840 real: 0.733642 / fake: 0.89515
    860 real: 0.734528 / fake: 0.894733
    880 real: 0.735361 / fake: 0.898804
    900 real: 0.736169 / fake: 0.886986
    920 real: 0.737327 / fake: 0.89359
    940 real: 0.737915 / fake: 0.891311
    960 real: 0.739335 / fake: 0.895348
    980 real: 0.740442 / fake: 0.887114
    1000 real: 0.740885 / fake: 0.893013
    1020 real: 0.741574 / fake: 0.892503
    1040 real: 0.742164 / fake: 0.897945
    1060 real: 0.743449 / fake: 0.891935
    1080 real: 0.74441 / fake: 0.889136
    1100 real: 0.745362 / fake: 0.889487
    1120 real: 0.747131 / fake: 0.881521
    1140 real: 0.748507 / fake: 0.89659
    1160 real: 0.751231 / fake: 0.89007
    1180 real: 0.752996 / fake: 0.887399
    1200 real: 0.754866 / fake: 0.89247
    1220 real: 0.755086 / fake: 0.887407
    1240 real: 0.755847 / fake: 0.867482
    1260 real: 0.757646 / fake: 0.843215
    1280 real: 0.756905 / fake: 0.819715
    1300 real: 0.75767 / fake: 0.803907
    1320 real: 0.757203 / fake: 0.827583
    1340 real: 0.757308 / fake: 0.850315
    1360 real: 0.757433 / fake: 0.871463
    1380 real: 0.757351 / fake: 0.889819
    1400 real: 0.757816 / fake: 0.893933
    1420 real: 0.757781 / fake: 0.888145
    1440 real: 0.75903 / fake: 0.892959
    1460 real: 0.759539 / fake: 0.883995
    1480 real: 0.760667 / fake: 0.887579
    1500 real: 0.761159 / fake: 0.870748
    1520 real: 0.762153 / fake: 0.832975
    1540 real: 0.76299 / fake: 0.786784
    1560 real: 0.763555 / fake: 0.8521
    1580 real: 0.765093 / fake: 0.943464
    1600 real: 0.767872 / fake: 0.868998
    1620 real: 0.770971 / fake: 0.841195
    1640 real: 0.77464 / fake: 0.770293
    1660 real: 0.77653 / fake: 0.915198
    1680 real: 0.779629 / fake: 0.894876
    1700 real: 0.781634 / fake: 0.868192
    1720 real: 0.782301 / fake: 0.895758
    1740 real: 0.782741 / fake: 0.892693
    1760 real: 0.783331 / fake: 0.883797
    1780 real: 0.783451 / fake: 0.889503
    1800 real: 0.783905 / fake: 0.890417
    1820 real: 0.783458 / fake: 0.893978
    1840 real: 0.783965 / fake: 0.886724
    1860 real: 0.784836 / fake: 0.887726
    1880 real: 0.784685 / fake: 0.882164
    1900 real: 0.785503 / fake: 0.885167
    1920 real: 0.785764 / fake: 0.891887
    1940 real: 0.786551 / fake: 0.858171
    1960 real: 0.78675 / fake: 0.820459
    1980 real: 0.786842 / fake: 0.747439
    2000 real: 0.786423 / fake: 0.684985
    2020 real: 0.785658 / fake: 0.661428
    2040 real: 0.784738 / fake: 0.68071
    2060 real: 0.783926 / fake: 0.732134
    2080 real: 0.78326 / fake: 0.763635
    2100 real: 0.78273 / fake: 0.756312
    2120 real: 0.782069 / fake: 0.763504
    2140 real: 0.781427 / fake: 0.730296
    2160 real: 0.780389 / fake: 0.682366
    2180 real: 0.779343 / fake: 0.639473
    2200 real: 0.778036 / fake: 0.611255
    2220 real: 0.776564 / fake: 0.618755
    2240 real: 0.775087 / fake: 0.667135
    2260 real: 0.774239 / fake: 0.731187
    2280 real: 0.773952 / fake: 0.780537
    2300 real: 0.773938 / fake: 0.795092
    2320 real: 0.774738 / fake: 0.749701
    2340 real: 0.774822 / fake: 0.799203
    2360 real: 0.775257 / fake: 0.846831
    2380 real: 0.776171 / fake: 0.801537
    2400 real: 0.776639 / fake: 0.80466
    2420 real: 0.777086 / fake: 0.771092
    2440 real: 0.777368 / fake: 0.770914
    2460 real: 0.777751 / fake: 0.769227
    2480 real: 0.778261 / fake: 0.759964
    2500 real: 0.778791 / fake: 0.738954
    2520 real: 0.77924 / fake: 0.731793
    2540 real: 0.779066 / fake: 0.712541
    2560 real: 0.778994 / fake: 0.694878
    2580 real: 0.779152 / fake: 0.664656
    2600 real: 0.778266 / fake: 0.634317
    2620 real: 0.777898 / fake: 0.61531
    2640 real: 0.776988 / fake: 0.62029
    2660 real: 0.775787 / fake: 0.616124
    2680 real: 0.775516 / fake: 0.63052
    2700 real: 0.775401 / fake: 0.654863
    2720 real: 0.779131 / fake: 0.618614
    2740 real: 0.777388 / fake: 0.590223
    2760 real: 0.777166 / fake: 0.585331
    2780 real: 0.776122 / fake: 0.602427
    2800 real: 0.774614 / fake: 0.61316
    2820 real: 0.773723 / fake: 0.615938
    2840 real: 0.77176 / fake: 0.612331
    2860 real: 0.769305 / fake: 0.596604
    2880 real: 0.768101 / fake: 0.593347
    2900 real: 0.765088 / fake: 0.624309
    2920 real: 0.762576 / fake: 0.634763
    2940 real: 0.761537 / fake: 0.637486
    2960 real: 0.759327 / fake: 0.638875
    2980 real: 0.758841 / fake: 0.636796
    3000 real: 0.756727 / fake: 0.634733
    3020 real: 0.755156 / fake: 0.628158
    3040 real: 0.754727 / fake: 0.623611
    3060 real: 0.752967 / fake: 0.618958
    3080 real: 0.75188 / fake: 0.610418
    3100 real: 0.750656 / fake: 0.59569
    3120 real: 0.750102 / fake: 0.582536
    ---------- EPOCH: 1
    


![png](readme_imgs/output_50_1.png)


    Model saved in file: ././gan.ckpt-0
    20 real: 0.748105 / fake: 0.571429
    40 real: 0.745575 / fake: 0.562493
    60 real: 0.744192 / fake: 0.555845
    80 real: 0.741732 / fake: 0.548718
    100 real: 0.740911 / fake: 0.561288
    120 real: 0.739659 / fake: 0.574443
    140 real: 0.738593 / fake: 0.588083
    160 real: 0.7361 / fake: 0.592384
    180 real: 0.735778 / fake: 0.598881
    200 real: 0.733334 / fake: 0.605741
    220 real: 0.733023 / fake: 0.614179
    240 real: 0.730922 / fake: 0.628962
    260 real: 0.729902 / fake: 0.639781
    280 real: 0.729569 / fake: 0.645844
    300 real: 0.728194 / fake: 0.646188
    320 real: 0.727431 / fake: 0.643623
    340 real: 0.726797 / fake: 0.638685
    360 real: 0.725343 / fake: 0.635014
    380 real: 0.724822 / fake: 0.631785
    400 real: 0.724202 / fake: 0.632701
    420 real: 0.723454 / fake: 0.633952
    440 real: 0.722311 / fake: 0.635574
    460 real: 0.7212 / fake: 0.638718
    480 real: 0.720479 / fake: 0.641427
    500 real: 0.719855 / fake: 0.64476
    520 real: 0.71876 / fake: 0.64907
    540 real: 0.718001 / fake: 0.653407
    560 real: 0.717029 / fake: 0.657251
    580 real: 0.716347 / fake: 0.659567
    600 real: 0.715535 / fake: 0.661183
    620 real: 0.714913 / fake: 0.662577
    640 real: 0.713999 / fake: 0.664387
    660 real: 0.713619 / fake: 0.666305
    680 real: 0.712768 / fake: 0.668087
    700 real: 0.712129 / fake: 0.669605
    720 real: 0.711588 / fake: 0.671041
    740 real: 0.710786 / fake: 0.67311
    760 real: 0.710523 / fake: 0.674719
    780 real: 0.709922 / fake: 0.675949
    800 real: 0.709487 / fake: 0.679999
    820 real: 0.709285 / fake: 0.706075
    840 real: 0.709052 / fake: 0.72
    860 real: 0.709126 / fake: 0.722446
    880 real: 0.709438 / fake: 0.720162
    900 real: 0.709815 / fake: 0.716742
    920 real: 0.710272 / fake: 0.714165
    940 real: 0.710303 / fake: 0.712306
    960 real: 0.710524 / fake: 0.711349
    980 real: 0.710692 / fake: 0.708492
    1000 real: 0.710676 / fake: 0.705184
    1020 real: 0.710813 / fake: 0.702184
    1040 real: 0.711102 / fake: 0.700062
    1060 real: 0.710828 / fake: 0.698304
    1080 real: 0.710732 / fake: 0.695981
    1100 real: 0.71067 / fake: 0.692815
    1120 real: 0.710584 / fake: 0.690133
    1140 real: 0.710452 / fake: 0.687266
    1160 real: 0.710375 / fake: 0.684621
    1180 real: 0.710222 / fake: 0.684801
    1200 real: 0.71045 / fake: 0.683784
    1220 real: 0.710108 / fake: 0.68255
    1240 real: 0.710018 / fake: 0.683197
    1260 real: 0.709734 / fake: 0.682892
    1280 real: 0.709537 / fake: 0.679607
    1300 real: 0.709738 / fake: 0.667557
    1320 real: 0.71036 / fake: 0.631387
    1340 real: 0.710953 / fake: 0.59317
    1360 real: 0.711158 / fake: 0.583154
    1380 real: 0.709985 / fake: 0.595112
    1400 real: 0.70931 / fake: 0.606511
    1420 real: 0.708331 / fake: 0.61705
    1440 real: 0.707629 / fake: 0.626936
    1460 real: 0.707442 / fake: 0.637972
    1480 real: 0.707685 / fake: 0.657751
    1500 real: 0.70539 / fake: 0.676003
    1520 real: 0.703442 / fake: 0.683064
    1540 real: 0.702221 / fake: 0.687683
    1560 real: 0.700997 / fake: 0.690064
    1580 real: 0.70005 / fake: 0.691618
    1600 real: 0.699054 / fake: 0.692186
    1620 real: 0.698436 / fake: 0.692113
    1640 real: 0.697904 / fake: 0.691313
    1660 real: 0.697306 / fake: 0.688154
    1680 real: 0.697571 / fake: 0.680833
    1700 real: 0.697002 / fake: 0.670441
    1720 real: 0.697473 / fake: 0.645575
    1740 real: 0.698108 / fake: 0.637125
    1760 real: 0.697866 / fake: 0.646399
    1780 real: 0.697565 / fake: 0.653257
    1800 real: 0.696672 / fake: 0.659842
    1820 real: 0.696389 / fake: 0.664454
    1840 real: 0.695067 / fake: 0.672564
    1860 real: 0.695146 / fake: 0.678529
    1880 real: 0.694719 / fake: 0.683128
    1900 real: 0.693815 / fake: 0.684896
    1920 real: 0.692893 / fake: 0.686272
    1940 real: 0.692534 / fake: 0.686761
    1960 real: 0.692223 / fake: 0.68655
    1980 real: 0.691853 / fake: 0.686362
    2000 real: 0.691485 / fake: 0.686056
    2020 real: 0.691244 / fake: 0.685531
    2040 real: 0.691301 / fake: 0.68531
    2060 real: 0.691339 / fake: 0.684074
    2080 real: 0.691392 / fake: 0.682062
    2100 real: 0.691412 / fake: 0.680031
    2120 real: 0.691503 / fake: 0.676666
    2140 real: 0.69182 / fake: 0.674799
    2160 real: 0.692091 / fake: 0.67352
    2180 real: 0.692897 / fake: 0.672385
    2200 real: 0.692539 / fake: 0.672405
    2220 real: 0.693263 / fake: 0.67234
    2240 real: 0.692938 / fake: 0.671234
    2260 real: 0.693197 / fake: 0.670231
    2280 real: 0.693721 / fake: 0.668688
    2300 real: 0.693935 / fake: 0.667798
    2320 real: 0.693698 / fake: 0.667954
    2340 real: 0.69381 / fake: 0.667723
    2360 real: 0.693588 / fake: 0.668543
    2380 real: 0.693056 / fake: 0.669123
    2400 real: 0.693221 / fake: 0.669856
    2420 real: 0.692252 / fake: 0.671385
    2440 real: 0.691944 / fake: 0.672486
    2460 real: 0.691339 / fake: 0.674561
    2480 real: 0.690824 / fake: 0.676248
    2500 real: 0.690141 / fake: 0.67863
    2520 real: 0.689128 / fake: 0.681333
    2540 real: 0.688682 / fake: 0.684675
    2560 real: 0.688002 / fake: 0.688079
    2580 real: 0.687377 / fake: 0.692195
    2600 real: 0.686536 / fake: 0.696538
    2620 real: 0.686323 / fake: 0.699755
    2640 real: 0.685611 / fake: 0.701544
    2660 real: 0.685188 / fake: 0.703059
    2680 real: 0.684847 / fake: 0.70432
    2700 real: 0.684851 / fake: 0.705285
    2720 real: 0.684659 / fake: 0.706053
    2740 real: 0.68504 / fake: 0.706629
    2760 real: 0.684828 / fake: 0.707147
    2780 real: 0.685206 / fake: 0.707881
    2800 real: 0.685637 / fake: 0.70826
    2820 real: 0.685901 / fake: 0.708738
    2840 real: 0.686376 / fake: 0.709342
    2860 real: 0.686891 / fake: 0.710332
    2880 real: 0.68726 / fake: 0.711214
    2900 real: 0.687489 / fake: 0.712427
    2920 real: 0.688426 / fake: 0.714138
    2940 real: 0.689118 / fake: 0.714694
    2960 real: 0.690347 / fake: 0.71477
    2980 real: 0.69122 / fake: 0.713266
    3000 real: 0.692028 / fake: 0.710555
    3020 real: 0.693185 / fake: 0.703147
    3040 real: 0.693839 / fake: 0.68646
    3060 real: 0.696144 / fake: 0.624766
    3080 real: 0.701882 / fake: 0.544824
    3100 real: 0.704527 / fake: 0.62168
    3120 real: 0.701702 / fake: 0.654818
    ---------- EPOCH: 2
    


![png](readme_imgs/output_50_3.png)


    Model saved in file: ././gan.ckpt-0
    20 real: 0.69911 / fake: 0.670672
    40 real: 0.696905 / fake: 0.678647
    60 real: 0.696166 / fake: 0.682311
    80 real: 0.695748 / fake: 0.685725
    100 real: 0.694163 / fake: 0.687823
    120 real: 0.693716 / fake: 0.689819
    140 real: 0.692657 / fake: 0.691791
    160 real: 0.691875 / fake: 0.6939
    180 real: 0.690534 / fake: 0.695509
    200 real: 0.689685 / fake: 0.69696
    220 real: 0.688813 / fake: 0.698068
    240 real: 0.688275 / fake: 0.698877
    260 real: 0.68813 / fake: 0.699501
    280 real: 0.687823 / fake: 0.699923
    300 real: 0.687412 / fake: 0.700201
    320 real: 0.687378 / fake: 0.700399
    340 real: 0.687279 / fake: 0.700474
    360 real: 0.687366 / fake: 0.700407
    380 real: 0.687298 / fake: 0.700302
    400 real: 0.687441 / fake: 0.700168
    420 real: 0.687562 / fake: 0.700026
    440 real: 0.687843 / fake: 0.699824
    460 real: 0.688057 / fake: 0.69964
    480 real: 0.688293 / fake: 0.699417
    500 real: 0.688502 / fake: 0.699229
    520 real: 0.688681 / fake: 0.699036
    540 real: 0.688997 / fake: 0.698845
    560 real: 0.689163 / fake: 0.698681
    580 real: 0.689313 / fake: 0.698487
    600 real: 0.689559 / fake: 0.69831
    620 real: 0.689822 / fake: 0.698152
    640 real: 0.689903 / fake: 0.698023
    660 real: 0.690168 / fake: 0.697835
    680 real: 0.690299 / fake: 0.697699
    700 real: 0.690526 / fake: 0.697569
    720 real: 0.690724 / fake: 0.69744
    740 real: 0.690908 / fake: 0.697348
    760 real: 0.6911 / fake: 0.697226
    780 real: 0.691246 / fake: 0.697123
    800 real: 0.691432 / fake: 0.697013
    820 real: 0.691573 / fake: 0.696991
    840 real: 0.691748 / fake: 0.696877
    860 real: 0.691784 / fake: 0.696815
    880 real: 0.692028 / fake: 0.696796
    900 real: 0.692127 / fake: 0.696798
    920 real: 0.692229 / fake: 0.696761
    940 real: 0.692442 / fake: 0.696711
    960 real: 0.692588 / fake: 0.696773
    980 real: 0.692654 / fake: 0.696806
    1000 real: 0.692759 / fake: 0.696818
    1020 real: 0.692849 / fake: 0.696857
    1040 real: 0.693014 / fake: 0.697009
    1060 real: 0.693181 / fake: 0.697146
    1080 real: 0.693277 / fake: 0.69728
    1100 real: 0.693414 / fake: 0.6974
    1120 real: 0.693555 / fake: 0.697446
    1140 real: 0.693706 / fake: 0.697433
    1160 real: 0.693788 / fake: 0.697497
    1180 real: 0.694041 / fake: 0.697646
    1200 real: 0.694155 / fake: 0.69762
    1220 real: 0.694204 / fake: 0.697704
    1240 real: 0.694402 / fake: 0.697659
    1260 real: 0.694452 / fake: 0.697681
    1280 real: 0.694677 / fake: 0.697511
    1300 real: 0.694701 / fake: 0.697493
    1320 real: 0.694875 / fake: 0.697349
    1340 real: 0.694965 / fake: 0.697095
    1360 real: 0.695037 / fake: 0.696772
    1380 real: 0.695314 / fake: 0.696418
    1400 real: 0.695343 / fake: 0.695836
    1420 real: 0.695525 / fake: 0.695188
    1440 real: 0.695566 / fake: 0.694598
    1460 real: 0.695634 / fake: 0.693784
    1480 real: 0.695887 / fake: 0.692757
    1500 real: 0.695987 / fake: 0.691568
    1520 real: 0.695971 / fake: 0.69009
    1540 real: 0.696476 / fake: 0.689362
    1560 real: 0.696131 / fake: 0.689838
    1580 real: 0.696011 / fake: 0.690162
    1600 real: 0.695772 / fake: 0.690108
    1620 real: 0.696087 / fake: 0.689856
    1640 real: 0.696054 / fake: 0.690053
    1660 real: 0.695738 / fake: 0.690641
    1680 real: 0.694846 / fake: 0.691659
    1700 real: 0.694791 / fake: 0.692277
    1720 real: 0.694164 / fake: 0.69269
    1740 real: 0.693933 / fake: 0.693047
    1760 real: 0.693842 / fake: 0.693342
    1780 real: 0.693496 / fake: 0.693566
    1800 real: 0.69343 / fake: 0.693826
    1820 real: 0.692921 / fake: 0.69399
    1840 real: 0.692848 / fake: 0.694108
    1860 real: 0.692777 / fake: 0.694162
    1880 real: 0.692844 / fake: 0.694226
    1900 real: 0.692795 / fake: 0.694245
    1920 real: 0.692736 / fake: 0.69426
    1940 real: 0.692874 / fake: 0.694255
    1960 real: 0.692785 / fake: 0.69424
    1980 real: 0.692886 / fake: 0.694217
    2000 real: 0.692884 / fake: 0.694187
    2020 real: 0.692892 / fake: 0.694152
    2040 real: 0.693043 / fake: 0.694127
    2060 real: 0.692967 / fake: 0.694076
    2080 real: 0.693073 / fake: 0.694046
    2100 real: 0.693116 / fake: 0.694013
    2120 real: 0.69313 / fake: 0.693959
    2140 real: 0.693206 / fake: 0.693914
    2160 real: 0.693164 / fake: 0.693854
    2180 real: 0.693229 / fake: 0.693817
    2200 real: 0.693262 / fake: 0.693771
    2220 real: 0.693292 / fake: 0.693739
    2240 real: 0.693317 / fake: 0.693697
    2260 real: 0.693287 / fake: 0.693661
    2280 real: 0.693367 / fake: 0.693617
    2300 real: 0.693365 / fake: 0.693571
    2320 real: 0.693404 / fake: 0.693528
    2340 real: 0.693374 / fake: 0.69348
    2360 real: 0.69338 / fake: 0.693443
    2380 real: 0.693468 / fake: 0.693388
    2400 real: 0.693445 / fake: 0.693346
    2420 real: 0.693428 / fake: 0.693308
    2440 real: 0.69342 / fake: 0.693263
    2460 real: 0.693553 / fake: 0.693207
    2480 real: 0.693511 / fake: 0.693142
    2500 real: 0.693564 / fake: 0.693047
    2520 real: 0.693639 / fake: 0.692813
    2540 real: 0.694234 / fake: 0.691993
    2560 real: 0.694727 / fake: 0.691478
    2580 real: 0.694432 / fake: 0.691655
    2600 real: 0.69433 / fake: 0.691828
    2620 real: 0.694271 / fake: 0.691983
    2640 real: 0.694069 / fake: 0.692121
    2660 real: 0.69379 / fake: 0.692361
    2680 real: 0.693656 / fake: 0.692487
    2700 real: 0.693605 / fake: 0.692538
    2720 real: 0.693569 / fake: 0.692558
    2740 real: 0.693468 / fake: 0.69265
    2760 real: 0.693299 / fake: 0.69277
    2780 real: 0.693326 / fake: 0.692818
    2800 real: 0.693323 / fake: 0.692774
    2820 real: 0.693404 / fake: 0.692766
    2840 real: 0.693404 / fake: 0.69281
    2860 real: 0.693237 / fake: 0.693037
    2880 real: 0.692946 / fake: 0.69329
    2900 real: 0.692773 / fake: 0.693588
    2920 real: 0.692648 / fake: 0.693735
    2940 real: 0.69264 / fake: 0.693788
    2960 real: 0.692697 / fake: 0.693796
    2980 real: 0.692753 / fake: 0.693769
    3000 real: 0.692793 / fake: 0.69373
    3020 real: 0.692838 / fake: 0.693706
    3040 real: 0.693013 / fake: 0.693687
    3060 real: 0.693011 / fake: 0.693671
    3080 real: 0.692995 / fake: 0.693685
    3100 real: 0.692987 / fake: 0.693673
    3120 real: 0.693174 / fake: 0.693667
    ---------- EPOCH: 3
    


![png](readme_imgs/output_50_5.png)


    Model saved in file: ././gan.ckpt-0
    20 real: 0.693197 / fake: 0.693654
    40 real: 0.693195 / fake: 0.693662
    60 real: 0.693235 / fake: 0.693655
    80 real: 0.693254 / fake: 0.693647
    100 real: 0.693318 / fake: 0.693638
    120 real: 0.693375 / fake: 0.693635
    140 real: 0.693399 / fake: 0.693651
    160 real: 0.693397 / fake: 0.693661
    180 real: 0.693461 / fake: 0.693668
    200 real: 0.69349 / fake: 0.693648
    220 real: 0.693459 / fake: 0.693694
    240 real: 0.693499 / fake: 0.693684
    260 real: 0.693505 / fake: 0.693672
    280 real: 0.693568 / fake: 0.693659
    300 real: 0.693582 / fake: 0.693632
    320 real: 0.693565 / fake: 0.693641
    340 real: 0.693585 / fake: 0.693637
    360 real: 0.693606 / fake: 0.69364
    380 real: 0.693604 / fake: 0.693587
    400 real: 0.693663 / fake: 0.693543
    420 real: 0.693637 / fake: 0.693482
    440 real: 0.693689 / fake: 0.693442
    460 real: 0.693679 / fake: 0.693429
    480 real: 0.693731 / fake: 0.693381
    500 real: 0.69379 / fake: 0.693292
    520 real: 0.693776 / fake: 0.693281
    540 real: 0.693575 / fake: 0.693434
    560 real: 0.693522 / fake: 0.693538
    580 real: 0.693507 / fake: 0.69354
    600 real: 0.693517 / fake: 0.693604
    620 real: 0.693379 / fake: 0.693721
    640 real: 0.693347 / fake: 0.693828
    660 real: 0.69341 / fake: 0.693879
    680 real: 0.69346 / fake: 0.693861
    700 real: 0.693562 / fake: 0.693878
    720 real: 0.693605 / fake: 0.693946
    740 real: 0.693606 / fake: 0.69402
    760 real: 0.693708 / fake: 0.694026
    780 real: 0.6938 / fake: 0.694004
    800 real: 0.693807 / fake: 0.693988
    820 real: 0.693831 / fake: 0.693911
    840 real: 0.693918 / fake: 0.693844
    860 real: 0.69398 / fake: 0.69373
    880 real: 0.693916 / fake: 0.693619
    900 real: 0.693839 / fake: 0.693611
    920 real: 0.693737 / fake: 0.693558
    940 real: 0.69378 / fake: 0.693429
    960 real: 0.693846 / fake: 0.693212
    980 real: 0.693779 / fake: 0.693172
    1000 real: 0.693556 / fake: 0.693247
    1020 real: 0.693447 / fake: 0.693131
    1040 real: 0.693642 / fake: 0.692864
    1060 real: 0.693782 / fake: 0.692633
    1080 real: 0.694221 / fake: 0.691995
    1100 real: 0.694518 / fake: 0.691507
    1120 real: 0.694458 / fake: 0.691463
    1140 real: 0.694457 / fake: 0.691576
    1160 real: 0.693951 / fake: 0.692112
    1180 real: 0.693433 / fake: 0.692588
    1200 real: 0.69345 / fake: 0.692514
    1220 real: 0.693614 / fake: 0.692298
    1240 real: 0.693545 / fake: 0.692348
    1260 real: 0.693563 / fake: 0.692396
    1280 real: 0.693659 / fake: 0.692311
    1300 real: 0.693592 / fake: 0.692577
    1320 real: 0.692714 / fake: 0.693705
    1340 real: 0.691571 / fake: 0.694768
    1360 real: 0.691435 / fake: 0.69484
    1380 real: 0.69197 / fake: 0.694499
    1400 real: 0.692302 / fake: 0.694248
    1420 real: 0.692243 / fake: 0.694238
    1440 real: 0.692541 / fake: 0.694033
    1460 real: 0.692858 / fake: 0.69378
    1480 real: 0.693056 / fake: 0.693624
    1500 real: 0.69316 / fake: 0.693568
    1520 real: 0.693276 / fake: 0.693527
    1540 real: 0.693313 / fake: 0.69351
    1560 real: 0.693319 / fake: 0.693527
    1580 real: 0.693244 / fake: 0.693579
    1600 real: 0.693259 / fake: 0.69358
    1620 real: 0.693296 / fake: 0.693551
    1640 real: 0.693363 / fake: 0.693482
    1660 real: 0.693432 / fake: 0.693404
    1680 real: 0.693503 / fake: 0.693305
    1700 real: 0.693503 / fake: 0.693287
    1720 real: 0.693457 / fake: 0.693306
    1740 real: 0.693506 / fake: 0.693309
    1760 real: 0.693426 / fake: 0.693271
    1780 real: 0.693401 / fake: 0.693219
    1800 real: 0.69343 / fake: 0.693214
    1820 real: 0.693336 / fake: 0.69326
    1840 real: 0.693255 / fake: 0.69338
    1860 real: 0.69311 / fake: 0.693526
    1880 real: 0.693071 / fake: 0.693581
    1900 real: 0.693128 / fake: 0.693546
    1920 real: 0.693245 / fake: 0.693525
    1940 real: 0.693305 / fake: 0.693516
    1960 real: 0.693344 / fake: 0.693552
    1980 real: 0.693336 / fake: 0.693596
    2000 real: 0.693391 / fake: 0.693633
    2020 real: 0.69341 / fake: 0.693653
    2040 real: 0.693448 / fake: 0.693646
    2060 real: 0.69347 / fake: 0.693632
    2080 real: 0.693461 / fake: 0.693591
    2100 real: 0.693522 / fake: 0.693519
    2120 real: 0.693495 / fake: 0.693439
    2140 real: 0.693519 / fake: 0.693309
    2160 real: 0.693498 / fake: 0.69314
    2180 real: 0.693523 / fake: 0.692857
    2200 real: 0.693729 / fake: 0.692506
    2220 real: 0.694693 / fake: 0.691634
    2240 real: 0.694385 / fake: 0.692493
    2260 real: 0.692551 / fake: 0.694151
    2280 real: 0.692359 / fake: 0.694297
    2300 real: 0.692695 / fake: 0.693942
    2320 real: 0.692999 / fake: 0.693644
    2340 real: 0.693159 / fake: 0.693482
    2360 real: 0.693217 / fake: 0.6934
    2380 real: 0.693268 / fake: 0.693356
    2400 real: 0.693257 / fake: 0.693336
    2420 real: 0.693271 / fake: 0.693328
    2440 real: 0.693279 / fake: 0.693323
    2460 real: 0.693259 / fake: 0.693325
    2480 real: 0.693257 / fake: 0.693304
    2500 real: 0.693254 / fake: 0.693285
    2520 real: 0.693248 / fake: 0.693264
    2540 real: 0.693248 / fake: 0.693236
    2560 real: 0.693248 / fake: 0.693209
    2580 real: 0.693222 / fake: 0.693175
    2600 real: 0.693204 / fake: 0.693123
    2620 real: 0.693201 / fake: 0.69308
    2640 real: 0.693183 / fake: 0.693041
    2660 real: 0.693143 / fake: 0.693005
    2680 real: 0.693133 / fake: 0.692942
    2700 real: 0.69321 / fake: 0.692844
    2720 real: 0.693667 / fake: 0.691974
    2740 real: 0.694491 / fake: 0.691028
    2760 real: 0.694863 / fake: 0.691266
    2780 real: 0.693404 / fake: 0.692958
    2800 real: 0.692636 / fake: 0.69399
    2820 real: 0.692173 / fake: 0.694487
    2840 real: 0.692141 / fake: 0.69452
    2860 real: 0.692393 / fake: 0.694221
    2880 real: 0.692794 / fake: 0.693801
    2900 real: 0.693134 / fake: 0.693517
    2920 real: 0.693306 / fake: 0.693412
    2940 real: 0.693321 / fake: 0.693424
    2960 real: 0.693322 / fake: 0.693501
    2980 real: 0.693294 / fake: 0.693551
    3000 real: 0.693348 / fake: 0.693527
    3020 real: 0.693421 / fake: 0.693523
    3040 real: 0.693463 / fake: 0.693498
    3060 real: 0.69349 / fake: 0.693497
    3080 real: 0.69351 / fake: 0.693488
    3100 real: 0.693491 / fake: 0.693485
    3120 real: 0.693523 / fake: 0.693481
    ---------- EPOCH: 4
    


![png](readme_imgs/output_50_7.png)


    Model saved in file: ././gan.ckpt-0
    20 real: 0.693472 / fake: 0.693439
    40 real: 0.693458 / fake: 0.693396
    60 real: 0.69343 / fake: 0.693345
    80 real: 0.693433 / fake: 0.693288
    100 real: 0.693376 / fake: 0.693186
    120 real: 0.693631 / fake: 0.692689
    140 real: 0.693956 / fake: 0.692303
    160 real: 0.693022 / fake: 0.693482
    180 real: 0.692478 / fake: 0.693929
    200 real: 0.692844 / fake: 0.693615
    220 real: 0.693094 / fake: 0.693378
    240 real: 0.69318 / fake: 0.693299
    260 real: 0.693195 / fake: 0.693298
    280 real: 0.69319 / fake: 0.693311
    300 real: 0.693212 / fake: 0.693312
    320 real: 0.693205 / fake: 0.693314
    340 real: 0.693208 / fake: 0.693313
    360 real: 0.693231 / fake: 0.693313
    380 real: 0.693235 / fake: 0.693308
    400 real: 0.693241 / fake: 0.693302
    420 real: 0.693233 / fake: 0.693291
    440 real: 0.693225 / fake: 0.693276
    460 real: 0.693218 / fake: 0.693261
    480 real: 0.69322 / fake: 0.693244
    500 real: 0.693212 / fake: 0.69323
    520 real: 0.693199 / fake: 0.693219
    540 real: 0.693186 / fake: 0.693203
    560 real: 0.693191 / fake: 0.693191
    580 real: 0.693173 / fake: 0.693175
    600 real: 0.693169 / fake: 0.693155
    620 real: 0.693215 / fake: 0.69314
    640 real: 0.693171 / fake: 0.693134
    660 real: 0.693173 / fake: 0.693124
    680 real: 0.693166 / fake: 0.693122
    700 real: 0.693166 / fake: 0.693125
    720 real: 0.693154 / fake: 0.693129
    740 real: 0.693152 / fake: 0.693138
    760 real: 0.693158 / fake: 0.693143
    780 real: 0.693155 / fake: 0.693151
    800 real: 0.69316 / fake: 0.693156
    820 real: 0.69317 / fake: 0.693161
    840 real: 0.693173 / fake: 0.693171
    860 real: 0.693176 / fake: 0.693187
    880 real: 0.693187 / fake: 0.693206
    900 real: 0.693208 / fake: 0.693232
    920 real: 0.693226 / fake: 0.69325
    940 real: 0.693253 / fake: 0.693268
    960 real: 0.693263 / fake: 0.693287
    980 real: 0.693279 / fake: 0.693294
    1000 real: 0.693292 / fake: 0.693287
    1020 real: 0.693287 / fake: 0.693288
    1040 real: 0.693292 / fake: 0.693282
    1060 real: 0.693289 / fake: 0.693273
    1080 real: 0.693285 / fake: 0.693261
    1100 real: 0.693274 / fake: 0.693244
    1120 real: 0.693276 / fake: 0.693236
    1140 real: 0.693263 / fake: 0.693218
    1160 real: 0.693254 / fake: 0.693199
    1180 real: 0.693242 / fake: 0.693193
    1200 real: 0.69323 / fake: 0.693189
    1220 real: 0.693213 / fake: 0.693194
    1240 real: 0.693213 / fake: 0.693203
    1260 real: 0.693209 / fake: 0.693214
    1280 real: 0.693195 / fake: 0.693223
    1300 real: 0.693195 / fake: 0.693232
    1320 real: 0.693207 / fake: 0.693246
    1340 real: 0.693213 / fake: 0.693251
    1360 real: 0.693225 / fake: 0.693252
    1380 real: 0.693224 / fake: 0.693252
    1400 real: 0.693243 / fake: 0.693249
    1420 real: 0.693228 / fake: 0.693254
    1440 real: 0.693233 / fake: 0.693256
    1460 real: 0.693239 / fake: 0.693253
    1480 real: 0.693238 / fake: 0.693253
    1500 real: 0.693249 / fake: 0.693252
    1520 real: 0.693248 / fake: 0.693254
    1540 real: 0.693233 / fake: 0.693253
    1560 real: 0.693241 / fake: 0.693246
    1580 real: 0.693247 / fake: 0.693239
    1600 real: 0.693246 / fake: 0.693236
    1620 real: 0.69324 / fake: 0.693231
    1640 real: 0.693228 / fake: 0.693231
    1660 real: 0.693222 / fake: 0.693229
    1680 real: 0.693215 / fake: 0.693223
    1700 real: 0.693212 / fake: 0.693217
    1720 real: 0.69321 / fake: 0.693214
    1740 real: 0.693198 / fake: 0.69321
    1760 real: 0.693193 / fake: 0.693205
    1780 real: 0.693203 / fake: 0.693197
    1800 real: 0.693187 / fake: 0.693194
    1820 real: 0.693185 / fake: 0.693189
    1840 real: 0.693186 / fake: 0.693183
    1860 real: 0.69318 / fake: 0.693177
    1880 real: 0.693176 / fake: 0.69318
    1900 real: 0.693179 / fake: 0.693173
    1920 real: 0.693167 / fake: 0.693176
    1940 real: 0.693167 / fake: 0.693171
    1960 real: 0.693168 / fake: 0.69317
    1980 real: 0.693165 / fake: 0.69317
    2000 real: 0.693164 / fake: 0.693169
    2020 real: 0.693165 / fake: 0.693171
    2040 real: 0.693179 / fake: 0.693176
    2060 real: 0.693172 / fake: 0.693179
    2080 real: 0.693182 / fake: 0.693183
    2100 real: 0.693179 / fake: 0.693193
    2120 real: 0.693191 / fake: 0.693199
    2140 real: 0.693189 / fake: 0.693207
    2160 real: 0.693193 / fake: 0.693198
    2180 real: 0.693193 / fake: 0.693194
    2200 real: 0.693187 / fake: 0.693188
    2220 real: 0.693181 / fake: 0.693184
    2240 real: 0.693178 / fake: 0.693179
    2260 real: 0.693174 / fake: 0.693175
    2280 real: 0.693172 / fake: 0.693177
    2300 real: 0.693172 / fake: 0.69318
    2320 real: 0.693172 / fake: 0.693189
    2340 real: 0.693178 / fake: 0.693195
    2360 real: 0.693182 / fake: 0.693201
    2380 real: 0.693192 / fake: 0.69321
    2400 real: 0.693201 / fake: 0.693221
    2420 real: 0.69321 / fake: 0.693232
    2440 real: 0.693213 / fake: 0.693241
    2460 real: 0.693226 / fake: 0.693245
    2480 real: 0.693223 / fake: 0.693245
    2500 real: 0.693225 / fake: 0.693243
    2520 real: 0.693228 / fake: 0.693236
    2540 real: 0.693229 / fake: 0.693228
    2560 real: 0.693224 / fake: 0.693218
    2580 real: 0.693221 / fake: 0.693205
    2600 real: 0.693214 / fake: 0.693195
    2620 real: 0.693207 / fake: 0.693185
    2640 real: 0.693199 / fake: 0.693183
    2660 real: 0.693191 / fake: 0.693182
    2680 real: 0.693182 / fake: 0.693183
    2700 real: 0.693192 / fake: 0.693183
    2720 real: 0.693186 / fake: 0.693177
    2740 real: 0.693184 / fake: 0.693181
    2760 real: 0.693186 / fake: 0.693185
    2780 real: 0.69319 / fake: 0.693188
    2800 real: 0.693194 / fake: 0.693194
    2820 real: 0.693193 / fake: 0.6932
    2840 real: 0.693197 / fake: 0.693203
    2860 real: 0.693199 / fake: 0.6932
    2880 real: 0.6932 / fake: 0.6932
    2900 real: 0.693192 / fake: 0.693204
    2920 real: 0.693194 / fake: 0.693202
    2940 real: 0.693189 / fake: 0.693201
    2960 real: 0.693191 / fake: 0.6932
    2980 real: 0.69319 / fake: 0.6932
    3000 real: 0.693189 / fake: 0.693195
    3020 real: 0.693182 / fake: 0.693198
    3040 real: 0.693179 / fake: 0.693198
    3060 real: 0.693181 / fake: 0.693192
    3080 real: 0.693172 / fake: 0.69319
    3100 real: 0.693176 / fake: 0.693185
    3120 real: 0.693173 / fake: 0.693179
    ---------- EPOCH: 5
    


![png](readme_imgs/output_50_9.png)


    Model saved in file: ././gan.ckpt-0
    20 real: 0.693174 / fake: 0.693178
    40 real: 0.693169 / fake: 0.693178
    60 real: 0.693174 / fake: 0.693178
    80 real: 0.69318 / fake: 0.693179
    100 real: 0.693182 / fake: 0.693183
    120 real: 0.693182 / fake: 0.693185
    140 real: 0.693184 / fake: 0.693188
    160 real: 0.693189 / fake: 0.693189
    180 real: 0.69319 / fake: 0.693193
    200 real: 0.693192 / fake: 0.693192
    220 real: 0.693196 / fake: 0.69319
    240 real: 0.693193 / fake: 0.69319
    260 real: 0.693191 / fake: 0.693187
    280 real: 0.693189 / fake: 0.693185
    300 real: 0.69318 / fake: 0.693181
    320 real: 0.693176 / fake: 0.693177
    340 real: 0.693172 / fake: 0.693171
    360 real: 0.693167 / fake: 0.693168
    380 real: 0.693164 / fake: 0.693168
    400 real: 0.693159 / fake: 0.693167
    420 real: 0.693161 / fake: 0.693168
    440 real: 0.693165 / fake: 0.693172
    460 real: 0.693167 / fake: 0.693173
    480 real: 0.693166 / fake: 0.693178
    500 real: 0.693167 / fake: 0.69318
    520 real: 0.693168 / fake: 0.693184
    540 real: 0.69317 / fake: 0.693186
    560 real: 0.693172 / fake: 0.693189
    580 real: 0.693177 / fake: 0.693189
    600 real: 0.693181 / fake: 0.693188
    620 real: 0.693178 / fake: 0.693189
    640 real: 0.693178 / fake: 0.693189
    660 real: 0.693181 / fake: 0.693189
    680 real: 0.693176 / fake: 0.69318
    700 real: 0.693168 / fake: 0.693179
    720 real: 0.693174 / fake: 0.693178
    740 real: 0.693177 / fake: 0.693173
    760 real: 0.693178 / fake: 0.693175
    780 real: 0.693173 / fake: 0.693176
    800 real: 0.693169 / fake: 0.693176
    820 real: 0.693167 / fake: 0.693177
    840 real: 0.693172 / fake: 0.693177
    860 real: 0.693167 / fake: 0.693179
    880 real: 0.69317 / fake: 0.693181
    900 real: 0.693186 / fake: 0.69318
    920 real: 0.693169 / fake: 0.693181
    940 real: 0.693172 / fake: 0.69318
    960 real: 0.693171 / fake: 0.693181
    980 real: 0.693172 / fake: 0.693182
    1000 real: 0.69317 / fake: 0.69318
    1020 real: 0.693167 / fake: 0.693181
    1040 real: 0.693164 / fake: 0.69318
    1060 real: 0.693162 / fake: 0.693175
    1080 real: 0.693159 / fake: 0.693173
    1100 real: 0.693159 / fake: 0.693169
    1120 real: 0.693161 / fake: 0.693165
    1140 real: 0.693159 / fake: 0.693166
    1160 real: 0.693165 / fake: 0.693165
    1180 real: 0.693165 / fake: 0.693166
    1200 real: 0.693167 / fake: 0.693167
    1220 real: 0.693168 / fake: 0.693169
    1240 real: 0.693172 / fake: 0.69317
    1260 real: 0.693172 / fake: 0.693173
    1280 real: 0.693173 / fake: 0.693176
    1300 real: 0.693173 / fake: 0.693178
    1320 real: 0.693166 / fake: 0.693184
    1340 real: 0.693163 / fake: 0.693183
    1360 real: 0.693164 / fake: 0.693184
    1380 real: 0.693162 / fake: 0.693185
    1400 real: 0.693161 / fake: 0.693182
    1420 real: 0.693158 / fake: 0.69318
    1440 real: 0.693164 / fake: 0.693174
    1460 real: 0.693164 / fake: 0.693172
    1480 real: 0.693164 / fake: 0.693169
    1500 real: 0.693166 / fake: 0.693167
    1520 real: 0.693166 / fake: 0.693167
    1540 real: 0.693165 / fake: 0.693164
    1560 real: 0.693167 / fake: 0.69316
    1580 real: 0.693159 / fake: 0.693163
    1600 real: 0.693163 / fake: 0.693166
    1620 real: 0.693155 / fake: 0.693175
    1640 real: 0.693142 / fake: 0.693179
    1660 real: 0.693158 / fake: 0.69317
    1680 real: 0.693159 / fake: 0.693168
    1700 real: 0.693155 / fake: 0.693166
    1720 real: 0.693172 / fake: 0.693162
    1740 real: 0.693167 / fake: 0.693147
    1760 real: 0.693162 / fake: 0.693163
    1780 real: 0.693156 / fake: 0.693168
    1800 real: 0.693158 / fake: 0.693166
    1820 real: 0.693162 / fake: 0.693164
    1840 real: 0.693161 / fake: 0.693163
    1860 real: 0.693159 / fake: 0.693165
    1880 real: 0.69316 / fake: 0.693164
    1900 real: 0.693163 / fake: 0.693164
    1920 real: 0.69316 / fake: 0.693157
    1940 real: 0.693163 / fake: 0.693163
    1960 real: 0.69316 / fake: 0.693169
    1980 real: 0.693157 / fake: 0.693171
    2000 real: 0.693155 / fake: 0.693171
    2020 real: 0.693159 / fake: 0.693169
    2040 real: 0.693155 / fake: 0.693169
    2060 real: 0.693153 / fake: 0.693168
    2080 real: 0.693154 / fake: 0.693166
    2100 real: 0.693155 / fake: 0.693162
    2120 real: 0.693159 / fake: 0.69316
    2140 real: 0.693171 / fake: 0.693156
    2160 real: 0.693159 / fake: 0.69315
    2180 real: 0.693164 / fake: 0.693149
    2200 real: 0.693159 / fake: 0.693159
    2220 real: 0.693153 / fake: 0.693165
    2240 real: 0.693154 / fake: 0.693167
    2260 real: 0.693152 / fake: 0.693168
    2280 real: 0.693152 / fake: 0.693166
    2300 real: 0.693153 / fake: 0.693164
    2320 real: 0.693156 / fake: 0.693162
    2340 real: 0.693157 / fake: 0.69316
    2360 real: 0.69317 / fake: 0.69316
    2380 real: 0.693158 / fake: 0.693158
    2400 real: 0.693157 / fake: 0.69316
    2420 real: 0.693156 / fake: 0.69316
    2440 real: 0.693156 / fake: 0.69316
    2460 real: 0.693155 / fake: 0.693161
    2480 real: 0.693155 / fake: 0.693161
    2500 real: 0.693158 / fake: 0.693158
    2520 real: 0.693156 / fake: 0.693161
    2540 real: 0.693155 / fake: 0.693161
    2560 real: 0.693156 / fake: 0.69316
    2580 real: 0.693156 / fake: 0.693158
    2600 real: 0.693158 / fake: 0.693156
    2620 real: 0.693156 / fake: 0.693157
    2640 real: 0.693156 / fake: 0.693157
    2660 real: 0.693155 / fake: 0.693158
    2680 real: 0.693155 / fake: 0.693158
    2700 real: 0.693152 / fake: 0.693159
    2720 real: 0.693151 / fake: 0.69316
    2740 real: 0.693151 / fake: 0.69316
    2760 real: 0.693151 / fake: 0.693161
    2780 real: 0.693152 / fake: 0.693161
    2800 real: 0.693153 / fake: 0.693159
    2820 real: 0.693154 / fake: 0.693157
    2840 real: 0.693155 / fake: 0.693156
    2860 real: 0.693155 / fake: 0.693156
    2880 real: 0.693155 / fake: 0.693156
    2900 real: 0.693155 / fake: 0.693157
    2920 real: 0.693155 / fake: 0.693157
    2940 real: 0.693153 / fake: 0.693159
    2960 real: 0.693153 / fake: 0.693159
    2980 real: 0.693156 / fake: 0.693157
    3000 real: 0.693155 / fake: 0.693156
    3020 real: 0.693156 / fake: 0.693156
    3040 real: 0.693157 / fake: 0.693154
    3060 real: 0.693157 / fake: 0.693153
    3080 real: 0.693157 / fake: 0.693153
    3100 real: 0.693154 / fake: 0.693154
    3120 real: 0.693154 / fake: 0.693154
    ---------- EPOCH: 6
    


![png](readme_imgs/output_50_11.png)


    Model saved in file: ././gan.ckpt-0
    20 real: 0.693152 / fake: 0.693154
    40 real: 0.693151 / fake: 0.693157
    60 real: 0.693151 / fake: 0.693156
    80 real: 0.69315 / fake: 0.693156
    100 real: 0.693153 / fake: 0.693156
    120 real: 0.693151 / fake: 0.693154
    140 real: 0.693152 / fake: 0.693153
    160 real: 0.693154 / fake: 0.693153
    180 real: 0.693153 / fake: 0.693154
    200 real: 0.693154 / fake: 0.693153
    220 real: 0.693153 / fake: 0.693153
    240 real: 0.693155 / fake: 0.693153
    260 real: 0.693153 / fake: 0.693154
    280 real: 0.693152 / fake: 0.693155
    300 real: 0.693151 / fake: 0.693156
    320 real: 0.693152 / fake: 0.693157
    340 real: 0.693149 / fake: 0.693158
    360 real: 0.693151 / fake: 0.693157
    380 real: 0.693152 / fake: 0.693156
    400 real: 0.693152 / fake: 0.693155
    420 real: 0.693153 / fake: 0.693154
    440 real: 0.693153 / fake: 0.693152
    460 real: 0.693156 / fake: 0.69315
    480 real: 0.693154 / fake: 0.69315
    500 real: 0.693152 / fake: 0.693152
    520 real: 0.693153 / fake: 0.693152
    540 real: 0.693151 / fake: 0.693153
    560 real: 0.693149 / fake: 0.693154
    580 real: 0.69315 / fake: 0.693154
    600 real: 0.693148 / fake: 0.693154
    620 real: 0.69315 / fake: 0.693153
    640 real: 0.693152 / fake: 0.693151
    660 real: 0.693155 / fake: 0.693149
    680 real: 0.693152 / fake: 0.69315
    700 real: 0.693152 / fake: 0.693151
    720 real: 0.693152 / fake: 0.693153
    740 real: 0.69315 / fake: 0.693155
    760 real: 0.693149 / fake: 0.693156
    780 real: 0.693147 / fake: 0.693157
    800 real: 0.69315 / fake: 0.693153
    820 real: 0.693152 / fake: 0.693151
    840 real: 0.693151 / fake: 0.693152
    860 real: 0.693151 / fake: 0.693152
    880 real: 0.69315 / fake: 0.693152
    900 real: 0.693149 / fake: 0.693152
    920 real: 0.693148 / fake: 0.693153
    940 real: 0.693148 / fake: 0.693153
    960 real: 0.693149 / fake: 0.693153
    980 real: 0.693151 / fake: 0.69315
    1000 real: 0.693153 / fake: 0.693149
    1020 real: 0.693154 / fake: 0.693149
    1040 real: 0.693152 / fake: 0.693151
    1060 real: 0.693152 / fake: 0.69315
    1080 real: 0.69315 / fake: 0.693152
    1100 real: 0.69315 / fake: 0.693153
    1120 real: 0.693149 / fake: 0.693153
    1140 real: 0.693151 / fake: 0.693152
    1160 real: 0.693149 / fake: 0.693151
    1180 real: 0.693149 / fake: 0.693152
    1200 real: 0.693149 / fake: 0.693151
    1220 real: 0.69315 / fake: 0.69315
    1240 real: 0.693149 / fake: 0.693151
    1260 real: 0.693147 / fake: 0.693153
    1280 real: 0.693151 / fake: 0.69315
    1300 real: 0.693151 / fake: 0.693149
    1320 real: 0.693151 / fake: 0.69315
    1340 real: 0.693149 / fake: 0.69315
    1360 real: 0.693148 / fake: 0.693152
    1380 real: 0.693148 / fake: 0.693152
    1400 real: 0.693149 / fake: 0.693151
    1420 real: 0.693152 / fake: 0.693149
    1440 real: 0.693153 / fake: 0.693147
    1460 real: 0.693152 / fake: 0.693149
    1480 real: 0.69315 / fake: 0.693149
    1500 real: 0.693148 / fake: 0.69315
    1520 real: 0.693146 / fake: 0.693153
    1540 real: 0.693146 / fake: 0.693153
    1560 real: 0.693151 / fake: 0.693149
    1580 real: 0.693154 / fake: 0.693145
    1600 real: 0.693151 / fake: 0.693148
    1620 real: 0.693152 / fake: 0.693147
    1640 real: 0.693148 / fake: 0.693149
    1660 real: 0.693146 / fake: 0.693152
    1680 real: 0.693145 / fake: 0.693156
    1700 real: 0.693145 / fake: 0.693155
    1720 real: 0.693149 / fake: 0.693151
    1740 real: 0.693151 / fake: 0.693148
    1760 real: 0.693155 / fake: 0.693145
    1780 real: 0.693153 / fake: 0.693145
    1800 real: 0.69315 / fake: 0.693149
    1820 real: 0.693149 / fake: 0.69315
    1840 real: 0.693148 / fake: 0.693151
    1860 real: 0.693147 / fake: 0.693151
    1880 real: 0.693158 / fake: 0.693139
    1900 real: 0.693143 / fake: 0.693154
    1920 real: 0.693149 / fake: 0.693149
    1940 real: 0.69315 / fake: 0.69315
    1960 real: 0.693148 / fake: 0.693149
    1980 real: 0.693147 / fake: 0.693148
    2000 real: 0.693149 / fake: 0.69315
    2020 real: 0.693149 / fake: 0.693154
    2040 real: 0.693148 / fake: 0.69315
    2060 real: 0.693145 / fake: 0.69315
    2080 real: 0.693147 / fake: 0.69315
    2100 real: 0.69315 / fake: 0.693149
    2120 real: 0.693151 / fake: 0.693148
    2140 real: 0.693152 / fake: 0.693145
    2160 real: 0.693151 / fake: 0.693145
    2180 real: 0.693153 / fake: 0.693146
    2200 real: 0.693146 / fake: 0.693152
    2220 real: 0.693145 / fake: 0.693151
    2240 real: 0.693144 / fake: 0.693153
    2260 real: 0.693142 / fake: 0.693155
    2280 real: 0.693143 / fake: 0.693151
    2300 real: 0.693147 / fake: 0.693149
    2320 real: 0.693151 / fake: 0.693147
    2340 real: 0.693154 / fake: 0.693145
    2360 real: 0.693158 / fake: 0.693136
    2380 real: 0.693143 / fake: 0.693143
    2400 real: 0.693166 / fake: 0.693131
    2420 real: 0.69315 / fake: 0.69315
    2440 real: 0.693147 / fake: 0.693153
    2460 real: 0.693145 / fake: 0.69315
    2480 real: 0.693145 / fake: 0.693148
    2500 real: 0.693146 / fake: 0.693148
    2520 real: 0.693148 / fake: 0.693148
    2540 real: 0.693151 / fake: 0.693149
    2560 real: 0.693152 / fake: 0.693148
    2580 real: 0.693151 / fake: 0.693147
    2600 real: 0.693149 / fake: 0.693146
    2620 real: 0.693148 / fake: 0.693146
    2640 real: 0.693148 / fake: 0.693149
    2660 real: 0.693148 / fake: 0.693151
    2680 real: 0.69315 / fake: 0.693151
    2700 real: 0.693149 / fake: 0.693149
    2720 real: 0.693145 / fake: 0.693149
    2740 real: 0.693144 / fake: 0.693148
    2760 real: 0.693146 / fake: 0.69315
    2780 real: 0.693149 / fake: 0.693153
    2800 real: 0.69315 / fake: 0.693152
    2820 real: 0.693146 / fake: 0.693148
    2840 real: 0.693147 / fake: 0.693146
    2860 real: 0.693151 / fake: 0.693149
    2880 real: 0.69315 / fake: 0.69315
    2900 real: 0.69315 / fake: 0.69315
    2920 real: 0.693151 / fake: 0.693147
    2940 real: 0.693133 / fake: 0.693155
    2960 real: 0.693146 / fake: 0.69315
    2980 real: 0.693148 / fake: 0.693148
    3000 real: 0.693148 / fake: 0.693147
    3020 real: 0.693148 / fake: 0.693146
    3040 real: 0.693148 / fake: 0.693147
    3060 real: 0.693147 / fake: 0.693146
    3080 real: 0.693145 / fake: 0.693145
    3100 real: 0.693145 / fake: 0.693145
    3120 real: 0.693146 / fake: 0.693147
    ---------- EPOCH: 7
    


![png](readme_imgs/output_50_13.png)


    Model saved in file: ././gan.ckpt-0
    20 real: 0.693151 / fake: 0.693151
    40 real: 0.693152 / fake: 0.693152
    60 real: 0.693149 / fake: 0.693149
    80 real: 0.693147 / fake: 0.693146
    100 real: 0.693147 / fake: 0.693148
    120 real: 0.69315 / fake: 0.693148
    140 real: 0.693152 / fake: 0.693147
    160 real: 0.693151 / fake: 0.69315
    180 real: 0.693147 / fake: 0.693148
    200 real: 0.693146 / fake: 0.693147
    220 real: 0.693146 / fake: 0.693146
    240 real: 0.693146 / fake: 0.693147
    260 real: 0.693147 / fake: 0.693148
    280 real: 0.693146 / fake: 0.693148
    300 real: 0.693146 / fake: 0.693146
    320 real: 0.693145 / fake: 0.693148
    340 real: 0.693147 / fake: 0.693151
    360 real: 0.693144 / fake: 0.693151
    380 real: 0.693149 / fake: 0.693149
    400 real: 0.693147 / fake: 0.693148
    420 real: 0.69315 / fake: 0.693149
    440 real: 0.693149 / fake: 0.693143
    460 real: 0.693148 / fake: 0.693145
    480 real: 0.693148 / fake: 0.693146
    500 real: 0.69315 / fake: 0.693149
    520 real: 0.693149 / fake: 0.693146
    540 real: 0.693146 / fake: 0.693146
    560 real: 0.693146 / fake: 0.693148
    580 real: 0.693149 / fake: 0.69315
    600 real: 0.693151 / fake: 0.693147
    620 real: 0.69315 / fake: 0.693149
    640 real: 0.693149 / fake: 0.693147
    660 real: 0.693147 / fake: 0.693147
    680 real: 0.693147 / fake: 0.693147
    700 real: 0.693146 / fake: 0.693148
    720 real: 0.693146 / fake: 0.693147
    740 real: 0.693146 / fake: 0.693146
    760 real: 0.693145 / fake: 0.693146
    780 real: 0.693147 / fake: 0.693148
    800 real: 0.693149 / fake: 0.693149
    820 real: 0.693152 / fake: 0.693142
    840 real: 0.693146 / fake: 0.693143
    860 real: 0.693144 / fake: 0.693148
    880 real: 0.693146 / fake: 0.693146
    900 real: 0.693146 / fake: 0.693149
    920 real: 0.693148 / fake: 0.693147
    940 real: 0.693152 / fake: 0.69315
    960 real: 0.69315 / fake: 0.693154
    980 real: 0.693157 / fake: 0.693151
    1000 real: 0.693147 / fake: 0.693147
    1020 real: 0.693147 / fake: 0.693147
    1040 real: 0.693147 / fake: 0.693147
    1060 real: 0.693146 / fake: 0.693148
    1080 real: 0.693146 / fake: 0.693148
    1100 real: 0.693147 / fake: 0.693147
    1120 real: 0.693146 / fake: 0.693146
    1140 real: 0.693146 / fake: 0.693144
    1160 real: 0.693145 / fake: 0.693143
    1180 real: 0.693143 / fake: 0.693144
    1200 real: 0.693142 / fake: 0.693144
    1220 real: 0.693144 / fake: 0.693147
    1240 real: 0.693149 / fake: 0.693152
    1260 real: 0.693168 / fake: 0.693163
    1280 real: 0.693156 / fake: 0.693154
    1300 real: 0.693151 / fake: 0.69315
    1320 real: 0.693146 / fake: 0.693141
    1340 real: 0.693134 / fake: 0.693154
    1360 real: 0.693214 / fake: 0.693239
    1380 real: 0.693134 / fake: 0.693162
    1400 real: 0.693149 / fake: 0.693146
    1420 real: 0.693151 / fake: 0.693145
    1440 real: 0.693155 / fake: 0.693142
    1460 real: 0.693154 / fake: 0.693149
    1480 real: 0.693154 / fake: 0.693155
    1500 real: 0.693153 / fake: 0.693153
    1520 real: 0.693148 / fake: 0.693153
    1540 real: 0.693148 / fake: 0.693148
    1560 real: 0.693144 / fake: 0.693146
    1580 real: 0.693143 / fake: 0.693146
    1600 real: 0.693144 / fake: 0.693145
    1620 real: 0.693148 / fake: 0.693146
    1640 real: 0.693145 / fake: 0.693147
    1660 real: 0.693147 / fake: 0.693146
    1680 real: 0.69315 / fake: 0.693144
    1700 real: 0.693146 / fake: 0.693144
    1720 real: 0.693152 / fake: 0.693147
    1740 real: 0.693152 / fake: 0.693152
    1760 real: 0.693152 / fake: 0.693153
    1780 real: 0.693154 / fake: 0.693153
    1800 real: 0.693158 / fake: 0.693158
    1820 real: 0.693167 / fake: 0.693171
    1840 real: 0.693142 / fake: 0.693144
    1860 real: 0.693135 / fake: 0.693136
    1880 real: 0.693141 / fake: 0.69314
    1900 real: 0.693144 / fake: 0.693142
    1920 real: 0.693146 / fake: 0.693145
    1940 real: 0.69315 / fake: 0.693149
    1960 real: 0.693155 / fake: 0.693155
    1980 real: 0.693146 / fake: 0.693145
    2000 real: 0.693154 / fake: 0.69315
    2020 real: 0.693166 / fake: 0.693163
    2040 real: 0.693154 / fake: 0.693154
    2060 real: 0.693152 / fake: 0.693152
    2080 real: 0.693149 / fake: 0.693152
    2100 real: 0.693146 / fake: 0.693148
    2120 real: 0.693142 / fake: 0.693141
    2140 real: 0.69315 / fake: 0.693147
    2160 real: 0.693147 / fake: 0.693144
    2180 real: 0.693148 / fake: 0.693145
    2200 real: 0.693146 / fake: 0.693148
    2220 real: 0.693147 / fake: 0.693147
    2240 real: 0.693148 / fake: 0.693148
    2260 real: 0.693149 / fake: 0.69315
    2280 real: 0.693152 / fake: 0.693152
    2300 real: 0.693153 / fake: 0.693154
    2320 real: 0.693155 / fake: 0.693156
    2340 real: 0.693166 / fake: 0.693156
    2360 real: 0.693155 / fake: 0.693156
    2380 real: 0.693136 / fake: 0.693139
    2400 real: 0.693146 / fake: 0.693135
    2420 real: 0.693131 / fake: 0.693106
    2440 real: 0.693154 / fake: 0.693148
    2460 real: 0.693177 / fake: 0.693196
    2480 real: 0.693121 / fake: 0.693118
    2500 real: 0.693149 / fake: 0.69315
    2520 real: 0.693159 / fake: 0.693155
    2540 real: 0.693178 / fake: 0.693169
    2560 real: 0.693167 / fake: 0.693152
    2580 real: 0.693158 / fake: 0.693155
    2600 real: 0.693146 / fake: 0.693152
    2620 real: 0.693148 / fake: 0.693146
    2640 real: 0.693134 / fake: 0.693125
    2660 real: 0.693144 / fake: 0.693133
    2680 real: 0.693163 / fake: 0.693158
    2700 real: 0.693158 / fake: 0.693157
    2720 real: 0.69315 / fake: 0.693148
    2740 real: 0.693157 / fake: 0.693153
    2760 real: 0.693148 / fake: 0.69315
    2780 real: 0.693152 / fake: 0.693152
    2800 real: 0.693158 / fake: 0.693155
    2820 real: 0.693169 / fake: 0.693161
    2840 real: 0.693158 / fake: 0.693147
    2860 real: 0.693153 / fake: 0.69316
    2880 real: 0.693161 / fake: 0.693159
    2900 real: 0.693156 / fake: 0.693159
    2920 real: 0.693154 / fake: 0.693157
    2940 real: 0.693151 / fake: 0.693148
    2960 real: 0.693147 / fake: 0.693142
    2980 real: 0.693138 / fake: 0.693138
    3000 real: 0.693136 / fake: 0.693125
    3020 real: 0.693137 / fake: 0.69312
    3040 real: 0.693143 / fake: 0.693137
    3060 real: 0.693148 / fake: 0.693164
    3080 real: 0.693149 / fake: 0.693147
    3100 real: 0.693151 / fake: 0.693151
    3120 real: 0.693154 / fake: 0.693155
    ---------- EPOCH: 8
    


![png](readme_imgs/output_50_15.png)


    Model saved in file: ././gan.ckpt-0
    20 real: 0.693181 / fake: 0.693177
    40 real: 0.693171 / fake: 0.69317
    60 real: 0.693168 / fake: 0.693172
    80 real: 0.693168 / fake: 0.693168
    100 real: 0.693158 / fake: 0.693161
    120 real: 0.69315 / fake: 0.693151
    140 real: 0.693124 / fake: 0.693113
    160 real: 0.693032 / fake: 0.692934
    180 real: 0.693226 / fake: 0.693167
    200 real: 0.693268 / fake: 0.693289
    220 real: 0.693193 / fake: 0.693182
    240 real: 0.693158 / fake: 0.693165
    260 real: 0.693164 / fake: 0.693165
    280 real: 0.693171 / fake: 0.693169
    300 real: 0.693171 / fake: 0.693169
    320 real: 0.693171 / fake: 0.693169
    340 real: 0.693162 / fake: 0.693153
    360 real: 0.693152 / fake: 0.693154
    380 real: 0.69317 / fake: 0.693161
    400 real: 0.693173 / fake: 0.693156
    420 real: 0.693138 / fake: 0.693131
    440 real: 0.693148 / fake: 0.693152
    460 real: 0.693152 / fake: 0.693156
    480 real: 0.693155 / fake: 0.693151
    500 real: 0.693152 / fake: 0.693154
    520 real: 0.693153 / fake: 0.693156
    540 real: 0.693153 / fake: 0.693153
    560 real: 0.693151 / fake: 0.693148
    580 real: 0.693154 / fake: 0.693154
    600 real: 0.693157 / fake: 0.69316
    620 real: 0.693154 / fake: 0.693152
    640 real: 0.693154 / fake: 0.693151
    660 real: 0.693153 / fake: 0.693151
    680 real: 0.693153 / fake: 0.693151
    700 real: 0.693151 / fake: 0.693153
    720 real: 0.693151 / fake: 0.693153
    740 real: 0.693151 / fake: 0.693153
    760 real: 0.693152 / fake: 0.693152
    780 real: 0.693151 / fake: 0.693152
    800 real: 0.693151 / fake: 0.69315
    820 real: 0.693151 / fake: 0.69315
    840 real: 0.693151 / fake: 0.693149
    860 real: 0.69315 / fake: 0.693149
    880 real: 0.69315 / fake: 0.693148
    900 real: 0.693149 / fake: 0.693149
    920 real: 0.693149 / fake: 0.693149
    940 real: 0.69315 / fake: 0.693148
    960 real: 0.693148 / fake: 0.693148
    980 real: 0.693149 / fake: 0.693149
    1000 real: 0.693151 / fake: 0.693148
    1020 real: 0.69315 / fake: 0.693147
    1040 real: 0.693148 / fake: 0.693149
    1060 real: 0.693146 / fake: 0.69315
    1080 real: 0.693149 / fake: 0.693147
    1100 real: 0.693147 / fake: 0.693149
    1120 real: 0.693149 / fake: 0.693148
    1140 real: 0.693147 / fake: 0.69315
    1160 real: 0.693148 / fake: 0.693148
    1180 real: 0.693148 / fake: 0.693148
    1200 real: 0.693147 / fake: 0.693148
    1220 real: 0.693147 / fake: 0.693149
    1240 real: 0.693147 / fake: 0.693148
    1260 real: 0.693147 / fake: 0.693148
    1280 real: 0.693147 / fake: 0.693148
    1300 real: 0.693148 / fake: 0.693147
    1320 real: 0.693147 / fake: 0.693148
    1340 real: 0.693147 / fake: 0.693148
    1360 real: 0.693147 / fake: 0.693148
    1380 real: 0.693146 / fake: 0.693148
    1400 real: 0.693146 / fake: 0.693147
    1420 real: 0.693146 / fake: 0.693147
    1440 real: 0.693148 / fake: 0.693146
    1460 real: 0.693147 / fake: 0.693146
    1480 real: 0.693147 / fake: 0.693145
    1500 real: 0.693146 / fake: 0.693147
    1520 real: 0.693147 / fake: 0.693147
    1540 real: 0.693147 / fake: 0.693148
    1560 real: 0.693149 / fake: 0.693147
    1580 real: 0.693149 / fake: 0.693147
    1600 real: 0.693149 / fake: 0.693148
    1620 real: 0.693149 / fake: 0.693148
    1640 real: 0.693147 / fake: 0.69315
    1660 real: 0.69315 / fake: 0.693148
    1680 real: 0.693149 / fake: 0.693148
    1700 real: 0.693149 / fake: 0.693149
    1720 real: 0.693148 / fake: 0.693151
    1740 real: 0.69315 / fake: 0.693148
    1760 real: 0.69315 / fake: 0.693149
    1780 real: 0.69315 / fake: 0.693149
    1800 real: 0.693149 / fake: 0.693149
    1820 real: 0.693149 / fake: 0.693149
    1840 real: 0.693148 / fake: 0.693151
    1860 real: 0.693151 / fake: 0.693148
    1880 real: 0.693149 / fake: 0.69315
    1900 real: 0.693151 / fake: 0.693148
    1920 real: 0.693148 / fake: 0.69315
    1940 real: 0.69315 / fake: 0.693148
    1960 real: 0.693148 / fake: 0.693149
    1980 real: 0.693149 / fake: 0.69315
    2000 real: 0.693149 / fake: 0.69315
    2020 real: 0.693149 / fake: 0.693149
    2040 real: 0.693151 / fake: 0.693148
    2060 real: 0.693148 / fake: 0.69315
    2080 real: 0.693147 / fake: 0.693151
    2100 real: 0.693152 / fake: 0.693146
    2120 real: 0.693147 / fake: 0.69315
    2140 real: 0.693149 / fake: 0.693148
    2160 real: 0.693149 / fake: 0.693148
    2180 real: 0.693147 / fake: 0.69315
    2200 real: 0.693148 / fake: 0.693149
    2220 real: 0.693148 / fake: 0.693148
    2240 real: 0.693149 / fake: 0.693148
    2260 real: 0.693147 / fake: 0.693149
    2280 real: 0.693146 / fake: 0.693149
    2300 real: 0.693149 / fake: 0.693146
    2320 real: 0.693148 / fake: 0.693147
    2340 real: 0.693147 / fake: 0.693148
    2360 real: 0.693147 / fake: 0.693148
    2380 real: 0.693147 / fake: 0.693148
    2400 real: 0.693147 / fake: 0.693147
    2420 real: 0.693147 / fake: 0.693147
    2440 real: 0.693145 / fake: 0.69315
    2460 real: 0.693152 / fake: 0.693144
    2480 real: 0.693145 / fake: 0.69315
    2500 real: 0.693146 / fake: 0.693149
    2520 real: 0.693151 / fake: 0.693143
    2540 real: 0.693143 / fake: 0.693151
    2560 real: 0.693147 / fake: 0.693147
    2580 real: 0.693148 / fake: 0.693148
    2600 real: 0.693148 / fake: 0.693148
    2620 real: 0.693147 / fake: 0.693148
    2640 real: 0.69315 / fake: 0.693144
    2660 real: 0.693143 / fake: 0.69315
    2680 real: 0.693147 / fake: 0.693148
    2700 real: 0.693148 / fake: 0.693147
    2720 real: 0.693151 / fake: 0.693145
    2740 real: 0.693148 / fake: 0.693148
    2760 real: 0.693147 / fake: 0.693148
    2780 real: 0.693147 / fake: 0.693147
    2800 real: 0.693148 / fake: 0.693146
    2820 real: 0.693147 / fake: 0.693146
    2840 real: 0.693147 / fake: 0.693147
    2860 real: 0.693147 / fake: 0.693148
    2880 real: 0.693148 / fake: 0.693148
    2900 real: 0.693146 / fake: 0.69315
    2920 real: 0.693148 / fake: 0.693147
    2940 real: 0.693148 / fake: 0.693147
    2960 real: 0.693147 / fake: 0.693147
    2980 real: 0.693147 / fake: 0.693147
    3000 real: 0.693148 / fake: 0.693148
    3020 real: 0.693148 / fake: 0.693147
    3040 real: 0.693148 / fake: 0.693148
    3060 real: 0.693147 / fake: 0.693147
    3080 real: 0.693146 / fake: 0.693148
    3100 real: 0.693146 / fake: 0.693148
    3120 real: 0.693146 / fake: 0.693147
    ---------- EPOCH: 9
    


![png](readme_imgs/output_50_17.png)


    Model saved in file: ././gan.ckpt-0
    20 real: 0.693147 / fake: 0.693147
    40 real: 0.693147 / fake: 0.693147
    60 real: 0.693147 / fake: 0.693148
    80 real: 0.693147 / fake: 0.693148
    100 real: 0.693147 / fake: 0.693148
    120 real: 0.693147 / fake: 0.693147
    140 real: 0.693147 / fake: 0.693147
    160 real: 0.693147 / fake: 0.693147
    180 real: 0.693147 / fake: 0.693147
    200 real: 0.693147 / fake: 0.693148
    220 real: 0.693147 / fake: 0.693148
    240 real: 0.693148 / fake: 0.693148
    260 real: 0.693147 / fake: 0.693148
    280 real: 0.693146 / fake: 0.693148
    300 real: 0.693147 / fake: 0.693147
    320 real: 0.693147 / fake: 0.693148
    340 real: 0.693148 / fake: 0.693147
    360 real: 0.693145 / fake: 0.693149
    380 real: 0.693148 / fake: 0.693146
    400 real: 0.693147 / fake: 0.693147
    420 real: 0.693147 / fake: 0.693146
    440 real: 0.693147 / fake: 0.693148
    460 real: 0.693151 / fake: 0.693141
    480 real: 0.693148 / fake: 0.693147
    500 real: 0.693145 / fake: 0.693152
    520 real: 0.693148 / fake: 0.693148
    540 real: 0.693147 / fake: 0.693145
    560 real: 0.693147 / fake: 0.693145
    580 real: 0.693151 / fake: 0.693143
    600 real: 0.693147 / fake: 0.693147
    620 real: 0.693148 / fake: 0.693148
    640 real: 0.693149 / fake: 0.693148
    660 real: 0.693147 / fake: 0.69315
    680 real: 0.693147 / fake: 0.693148
    700 real: 0.693148 / fake: 0.693146
    720 real: 0.693152 / fake: 0.693144
    740 real: 0.693147 / fake: 0.693146
    760 real: 0.693143 / fake: 0.69315
    780 real: 0.693148 / fake: 0.693145
    800 real: 0.693146 / fake: 0.693147
    820 real: 0.693147 / fake: 0.693147
    840 real: 0.693148 / fake: 0.693147
    860 real: 0.693149 / fake: 0.693147
    880 real: 0.693147 / fake: 0.693148
    900 real: 0.693147 / fake: 0.693148
    920 real: 0.693145 / fake: 0.693149
    940 real: 0.693148 / fake: 0.693144
    960 real: 0.693149 / fake: 0.693145
    980 real: 0.693146 / fake: 0.693149
    1000 real: 0.693145 / fake: 0.693149
    1020 real: 0.693146 / fake: 0.693147
    1040 real: 0.69315 / fake: 0.693144
    1060 real: 0.693143 / fake: 0.69315
    1080 real: 0.693147 / fake: 0.693148
    1100 real: 0.69315 / fake: 0.693145
    1120 real: 0.693145 / fake: 0.693151
    1140 real: 0.693149 / fake: 0.693148
    1160 real: 0.693146 / fake: 0.693148
    1180 real: 0.693142 / fake: 0.693149
    1200 real: 0.693149 / fake: 0.693145
    1220 real: 0.693148 / fake: 0.693146
    1240 real: 0.693146 / fake: 0.69315
    1260 real: 0.693148 / fake: 0.693147
    1280 real: 0.693145 / fake: 0.693149
    1300 real: 0.693146 / fake: 0.693146
    1320 real: 0.693147 / fake: 0.693146
    1340 real: 0.693148 / fake: 0.693146
    1360 real: 0.693146 / fake: 0.693146
    1380 real: 0.693149 / fake: 0.693147
    1400 real: 0.69315 / fake: 0.693155
    1420 real: 0.693148 / fake: 0.693133
    1440 real: 0.693154 / fake: 0.693156
    1460 real: 0.693132 / fake: 0.693126
    1480 real: 0.693181 / fake: 0.693132
    1500 real: 0.693132 / fake: 0.693138
    1520 real: 0.693141 / fake: 0.693112
    1540 real: 0.693165 / fake: 0.693179
    1560 real: 0.693164 / fake: 0.693134
    1580 real: 0.693154 / fake: 0.693146
    1600 real: 0.693164 / fake: 0.693156
    1620 real: 0.693161 / fake: 0.693156
    1640 real: 0.693149 / fake: 0.693144
    1660 real: 0.693162 / fake: 0.693157
    1680 real: 0.693147 / fake: 0.693135
    1700 real: 0.693149 / fake: 0.693146
    1720 real: 0.693146 / fake: 0.693145
    1740 real: 0.693147 / fake: 0.693145
    1760 real: 0.693148 / fake: 0.693146
    1780 real: 0.693151 / fake: 0.693149
    1800 real: 0.693147 / fake: 0.693147
    1820 real: 0.693147 / fake: 0.693147
    1840 real: 0.693147 / fake: 0.693148
    1860 real: 0.693145 / fake: 0.693147
    1880 real: 0.693149 / fake: 0.693147
    1900 real: 0.693147 / fake: 0.693149
    1920 real: 0.693148 / fake: 0.69315
    1940 real: 0.693148 / fake: 0.693151
    1960 real: 0.693151 / fake: 0.693149
    1980 real: 0.693149 / fake: 0.693152
    2000 real: 0.69315 / fake: 0.693151
    2020 real: 0.69315 / fake: 0.69315
    2040 real: 0.69315 / fake: 0.69315
    2060 real: 0.693151 / fake: 0.69315
    2080 real: 0.693154 / fake: 0.693151
    2100 real: 0.693151 / fake: 0.693152
    2120 real: 0.693151 / fake: 0.693145
    2140 real: 0.693149 / fake: 0.693151
    2160 real: 0.693151 / fake: 0.693151
    2180 real: 0.693153 / fake: 0.693152
    2200 real: 0.693151 / fake: 0.69315
    2220 real: 0.693149 / fake: 0.693146
    2240 real: 0.693145 / fake: 0.69315
    2260 real: 0.693146 / fake: 0.693154
    2280 real: 0.69315 / fake: 0.693149
    2300 real: 0.693146 / fake: 0.693152
    2320 real: 0.693149 / fake: 0.693151
    2340 real: 0.693152 / fake: 0.693149
    2360 real: 0.693148 / fake: 0.693147
    2380 real: 0.693146 / fake: 0.693146
    2400 real: 0.693147 / fake: 0.693146
    2420 real: 0.693148 / fake: 0.693147
    2440 real: 0.693148 / fake: 0.693148
    2460 real: 0.693148 / fake: 0.693148
    2480 real: 0.693148 / fake: 0.693148
    2500 real: 0.693149 / fake: 0.693147
    2520 real: 0.69315 / fake: 0.693145
    2540 real: 0.693147 / fake: 0.693148
    2560 real: 0.693147 / fake: 0.693148
    2580 real: 0.693148 / fake: 0.693147
    2600 real: 0.693147 / fake: 0.693148
    2620 real: 0.693148 / fake: 0.693148
    2640 real: 0.693147 / fake: 0.693149
    2660 real: 0.693148 / fake: 0.693148
    2680 real: 0.693146 / fake: 0.693149
    2700 real: 0.693148 / fake: 0.693148
    2720 real: 0.693148 / fake: 0.693148
    2740 real: 0.693149 / fake: 0.693148
    2760 real: 0.693148 / fake: 0.693148
    2780 real: 0.693148 / fake: 0.693148
    2800 real: 0.693148 / fake: 0.693148
    2820 real: 0.693148 / fake: 0.693147
    2840 real: 0.693147 / fake: 0.693147
    2860 real: 0.693147 / fake: 0.693148
    2880 real: 0.693148 / fake: 0.693147
    2900 real: 0.693151 / fake: 0.693146
    2920 real: 0.693149 / fake: 0.693148
    2940 real: 0.693149 / fake: 0.693148
    2960 real: 0.693149 / fake: 0.693147
    2980 real: 0.693146 / fake: 0.693149
    3000 real: 0.693147 / fake: 0.693147
    3020 real: 0.693147 / fake: 0.693147
    3040 real: 0.693148 / fake: 0.693146
    3060 real: 0.693147 / fake: 0.693148
    3080 real: 0.69315 / fake: 0.693145
    3100 real: 0.69315 / fake: 0.693145
    3120 real: 0.693147 / fake: 0.693147
    ---------- EPOCH: 10
    


![png](readme_imgs/output_50_19.png)


    Model saved in file: ././gan.ckpt-0
    


```python
# Tell all the threads to shutdown.
coord.request_stop()

# Wait until all threads have finished.
coord.join(threads)

# Clean up the session.
sess.close()
```

<a name="part-2---variational-auto-encoding-generative-adversarial-network-vaegan"></a>
# Part 2 - Variational Auto-Encoding Generative Adversarial Network (VAEGAN)

In our definition of the generator, we started with a feature vector, `Z`.  This feature vector was not connected to anything before it.  Instead, we had to randomly create its values using a random number generator of its `n_latent` values from -1 to 1, and this range was chosen arbitrarily.  It could have been 0 to 1, or -3 to 3, or 0 to 100.  In any case, the network would have had to learn to transform those values into something that looked like an image.  There was no way for us to take an image, and find the feature vector that created it.  In other words, it was not possible for us to *encode* an image.

The closest thing to an encoding we had was taking an image and feeding it to the discriminator, which would output a 0 or 1.  But what if we had another network that allowed us to encode an image, and then we used this network for both the discriminator and generative parts of the network?  That's the basic idea behind the VAEGAN: https://arxiv.org/abs/1512.09300.  It is just like the regular GAN, except we also use an encoder to create our feature vector `Z`.

We then get the best of both worlds: a GAN that looks more or less the same, but uses the encoding from an encoder instead of an arbitrary feature vector; and an autoencoder that can model an input distribution using a trained distance function, the discriminator, leading to nicer encodings/decodings.

Let's try to build it!  Refer to the paper for the intricacies and a great read.  Luckily, by building the `encoder` and `decoder` functions, we're almost there.  We just need a few more components and will change these slightly.

Let's reset our graph and recompose our network as a VAEGAN:


```python
tf.reset_default_graph()
```

<a name="batch-normalization"></a>
## Batch Normalization

You may have noticed from the `VAE` code that I've used something called "batch normalization".  This is a pretty effective technique for regularizing the training of networks by "reducing internal covariate shift".  The basic idea is that given a minibatch, we optimize the gradient for this small sample of the greater population.  But this small sample may have different characteristics than the entire population's gradient.  Consider the most extreme case, a minibatch of 1.  In this case, we overfit our gradient to optimize the gradient of the single observation.  If our minibatch is too large, say the size of the entire population, we aren't able to manuvuer the loss manifold at all and the entire loss is averaged in a way that doesn't let us optimize anything.  What we want to do is find a happy medium between a too-smooth loss surface (i.e. every observation), and a very peaky loss surface (i.e. a single observation).  Up until now we only used mini-batches to help with this.  But we can also approach it by "smoothing" our updates between each mini-batch.  That would effectively smooth the manifold of the loss space.  Those of you familiar with signal processing will see this as a sort of low-pass filter on the gradient updates.

In order for us to use batch normalization, we need another placeholder which is a simple boolean: True or False, denoting when we are training.  We'll use this placeholder to conditionally update batch normalization's statistics required for normalizing our minibatches.  Let's create the placeholder and then I'll get into how to use this.


```python
# placeholder for batch normalization
is_training = tf.placeholder(tf.bool, name='istraining')
```

The original paper that introduced the idea suggests to use batch normalization "pre-activation", meaning after the weight multipllication or convolution, and before the nonlinearity.  We can use the `tensorflow.contrib.layers.batch_norm` module to apply batch normalization to any input tensor give the tensor and the placeholder defining whether or not we are training.  Let's use this module and you can inspect the code inside the module in your own time if it interests you.


```python
from tensorflow.contrib.layers import batch_norm
help(batch_norm)
```

    Help on function batch_norm in module tensorflow.contrib.framework.python.ops.arg_scope:
    
    batch_norm(inputs, decay=0.999, center=True, scale=False, epsilon=0.001, activation_fn=None, param_initializers=None, param_regularizers=None, updates_collections='update_ops', is_training=True, reuse=None, variables_collections=None, outputs_collections=None, trainable=True, batch_weights=None, fused=False, data_format='NHWC', zero_debias_moving_mean=False, scope=None, renorm=False, renorm_clipping=None, renorm_decay=0.99)
        Adds a Batch Normalization layer from http://arxiv.org/abs/1502.03167.
        
          "Batch Normalization: Accelerating Deep Network Training by Reducing
          Internal Covariate Shift"
        
          Sergey Ioffe, Christian Szegedy
        
        Can be used as a normalizer function for conv2d and fully_connected.
        
        Note: when training, the moving_mean and moving_variance need to be updated.
        By default the update ops are placed in `tf.GraphKeys.UPDATE_OPS`, so they
        need to be added as a dependency to the `train_op`. For example:
        
        ```python
          update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
          with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss)
        ```
        
        One can set updates_collections=None to force the updates in place, but that
        can have a speed penalty, especially in distributed settings.
        
        Args:
          inputs: A tensor with 2 or more dimensions, where the first dimension has
            `batch_size`. The normalization is over all but the last dimension if
            `data_format` is `NHWC` and the second dimension if `data_format` is
            `NCHW`.
          decay: Decay for the moving average. Reasonable values for `decay` are close
            to 1.0, typically in the multiple-nines range: 0.999, 0.99, 0.9, etc.
            Lower `decay` value (recommend trying `decay`=0.9) if model experiences
            reasonably good training performance but poor validation and/or test
            performance. Try zero_debias_moving_mean=True for improved stability.
          center: If True, add offset of `beta` to normalized tensor. If False, `beta`
            is ignored.
          scale: If True, multiply by `gamma`. If False, `gamma` is
            not used. When the next layer is linear (also e.g. `nn.relu`), this can be
            disabled since the scaling can be done by the next layer.
          epsilon: Small float added to variance to avoid dividing by zero.
          activation_fn: Activation function, default set to None to skip it and
            maintain a linear activation.
          param_initializers: Optional initializers for beta, gamma, moving mean and
            moving variance.
          param_regularizers: Optional regularizer for beta and gamma.
          updates_collections: Collections to collect the update ops for computation.
            The updates_ops need to be executed with the train_op.
            If None, a control dependency would be added to make sure the updates are
            computed in place.
          is_training: Whether or not the layer is in training mode. In training mode
            it would accumulate the statistics of the moments into `moving_mean` and
            `moving_variance` using an exponential moving average with the given
            `decay`. When it is not in training mode then it would use the values of
            the `moving_mean` and the `moving_variance`.
          reuse: Whether or not the layer and its variables should be reused. To be
            able to reuse the layer scope must be given.
          variables_collections: Optional collections for the variables.
          outputs_collections: Collections to add the outputs.
          trainable: If `True` also add variables to the graph collection
            `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
          batch_weights: An optional tensor of shape `[batch_size]`,
            containing a frequency weight for each batch item. If present,
            then the batch normalization uses weighted mean and
            variance. (This can be used to correct for bias in training
            example selection.)
          fused:  Use nn.fused_batch_norm if True, nn.batch_normalization otherwise.
          data_format: A string. `NHWC` (default) and `NCHW` are supported.
          zero_debias_moving_mean: Use zero_debias for moving_mean. It creates a new
            pair of variables 'moving_mean/biased' and 'moving_mean/local_step'.
          scope: Optional scope for `variable_scope`.
          renorm: Whether to use Batch Renormalization
            (https://arxiv.org/abs/1702.03275). This adds extra variables during
            training. The inference is the same for either value of this parameter.
          renorm_clipping: A dictionary that may map keys 'rmax', 'rmin', 'dmax' to
            scalar `Tensors` used to clip the renorm correction. The correction
            `(r, d)` is used as `corrected_value = normalized_value * r + d`, with
            `r` clipped to [rmin, rmax], and `d` to [-dmax, dmax]. Missing rmax, rmin,
            dmax are set to inf, 0, inf, respectively.
          renorm_decay: Momentum used to update the moving means and standard
            deviations with renorm. Unlike `momentum`, this affects training
            and should be neither too small (which would add noise) nor too large
            (which would give stale estimates). Note that `decay` is still applied
            to get the means and variances for inference.
        
        Returns:
          A `Tensor` representing the output of the operation.
        
        Raises:
          ValueError: If `batch_weights` is not None and `fused` is True.
          ValueError: If `param_regularizers` is not None and `fused` is True.
          ValueError: If `data_format` is neither `NHWC` nor `NCHW`.
          ValueError: If the rank of `inputs` is undefined.
          ValueError: If rank or channels dimension of `inputs` is undefined.
    
    

<a name="building-the-encoder-1"></a>
## Building the Encoder

We can now change our encoder to accept the `is_training` placeholder and apply `batch_norm` just before the activation function is applied:


```python
def encoder(x, is_training, channels, filter_sizes, activation=tf.nn.tanh, reuse=None):
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
            print('layer:', layer_i, ', shape:', h.get_shape().as_list())
            
            # Store each hidden layer
            hs.append(h)

    # Finally, return the encoding.
    return h, hs
```

Let's now create the input to the network using a placeholder.  We can try a slightly larger image this time.  But be careful experimenting with much larger images as this is a big network.

<h3><font color='red'>TODO! COMPLETE THIS SECTION!</font></h3>


```python
n_pixels = 64
n_channels = 3
input_shape = [None, n_pixels, n_pixels, n_channels]

# placeholder for the input to the network
X = tf.placeholder(name='X', shape=input_shape, dtype=tf.float32)
```

And now we'll connect the input to an encoder network.  We'll also use the `tf.nn.elu` activation instead.  Explore other activations but I've found this to make the training much faster (e.g. 10x faster at least!).  See the paper for more details: [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
](http://arxiv.org/abs/1511.07289)

<h3><font color='red'>TODO! COMPLETE THIS SECTION!</font></h3>


```python
channels = [64, 64, 64]
filter_sizes = [5, 5, 5]
activation = tf.nn.elu
n_hidden = 128

with tf.variable_scope('encoder'):
    H, Hs = encoder(
                x=X,
                is_training=is_training,
                channels=channels,
                filter_sizes=filter_sizes,
                activation=activation,
                reuse=None)
    Z = utils.linear(H, n_hidden)[0]
```

    encoder/input: [None, 64, 64, 3]
    layer: 0 , shape: [None, 32, 32, 64]
    layer: 1 , shape: [None, 16, 16, 64]
    layer: 2 , shape: [None, 8, 8, 64]
    

<a name="building-the-variational-layer"></a>
## Building the Variational Layer

In Session 3, we introduced the idea of Variational Bayes when we used the Variational Auto Encoder.  The variational bayesian approach requires a richer understanding of probabilistic graphical models and bayesian methods which we we're not able to go over in this course (it requires a few courses all by itself!).  For that reason, please treat this as a "black box" in this course.

For those of you that are more familiar with graphical models, Variational Bayesian methods attempt to model an approximate joint distribution of $Q(Z)$ using some distance function to the true distribution $P(X)$.  Kingma and Welling show how this approach can be used in a graphical model resembling an autoencoder and can be trained using KL-Divergence, or $KL(Q(Z) || P(X))$.  The distribution Q(Z) is the variational distribution, and attempts to model the lower-bound of the true distribution $P(X)$ through the minimization of the KL-divergence.  Another way to look at this is the encoder of the network is trying to model the parameters of a known distribution, the Gaussian Distribution, through a minimization of this lower bound.  We assume that this distribution resembles the true distribution, but it is merely a simplification of the true distribution.  To learn more about this, I highly recommend picking up the book by Christopher Bishop called "Pattern Recognition and Machine Learning" and reading the original Kingma and Welling paper on Variational Bayes.

Now back to coding, we'll create a general variational layer that does exactly the same thing as our VAE in session 3.  Treat this as a black box if you are unfamiliar with the math.  It takes an input encoding, `h`, and an integer, `n_code` defining how many latent Gaussians to use to model the latent distribution.  In return, we get the latent encoding from sampling the Gaussian layer, `z`, the mean and log standard deviation, as well as the prior loss, `loss_z`.


```python
def variational_bayes(h, n_code):
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
```

Let's connect this layer to our encoding, and keep all the variables it returns.  Treat this as a black box if you are unfamiliar with variational bayes!

<h3><font color='red'>TODO! COMPLETE THIS SECTION!</font></h3>


```python
# Experiment w/ values between 2 - 100
# depending on how difficult the dataset is
n_code = 32

with tf.variable_scope('encoder/variational'):
    Z, Z_mu, Z_log_sigma, loss_Z = variational_bayes(h=Z, n_code=n_code)
    print('variational encoder shape:', Z.get_shape().as_list())
```

    variational encoder shape: [None, 32]
    

<a name="building-the-decoder-1"></a>
## Building the Decoder

In the GAN network, we built a decoder and called it the generator network.  Same idea here.  We can use these terms interchangeably.  Before we connect our latent encoding, `Z` to the decoder, we'll implement batch norm in our decoder just like we did with the encoder.  This is a simple fix: add a second argument for `is_training` and then apply batch normalization just after the `deconv2d` operation and just before the nonlinear `activation`.


```python
def decoder(z, is_training, dimensions, channels, filter_sizes,
            activation=tf.nn.elu, reuse=None):
    h = z
    for layer_i in range(len(dimensions)):
        with tf.variable_scope('layer{}'.format(layer_i+1), reuse=reuse):
            h, W = utils.deconv2d(x=h,
                               n_output_h=dimensions[layer_i],
                               n_output_w=dimensions[layer_i],
                               n_output_ch=channels[layer_i],
                               k_h=filter_sizes[layer_i],
                               k_w=filter_sizes[layer_i],
                               reuse=reuse)
            h = batch_norm(h, is_training=is_training)
            h = activation(h)
            print('layer:', layer_i, ', shape:', h.get_shape().as_list())
    return h
```

Now we'll build a decoder just like in Session 3, and just like our Generator network in Part 1.  In Part 1, we created `Z` as a placeholder which we would have had to feed in as random values.  However, now we have an explicit coding of an input image in `X` stored in `Z` by having created the encoder network.


```python
dimensions = [n_pixels // 8, n_pixels // 4, n_pixels // 2, n_pixels]
channels = [30, 30, 30, n_channels]
filter_sizes = [4, 4, 4, 4]
activation = tf.nn.elu
n_latent = n_code * (n_pixels // 16)**2

with tf.variable_scope('generator'):
    Z_decode = utils.linear(
        Z, n_output=n_latent, name='fc', activation=activation)[0]
    Z_decode_tensor = tf.reshape(
        Z_decode, [-1, n_pixels//16, n_pixels//16, n_code], name='reshape')
    G = decoder(
        Z_decode_tensor, is_training, dimensions,
        channels, filter_sizes, activation)
```

    layer: 0 , shape: [None, 8, 8, 30]
    layer: 1 , shape: [None, 16, 16, 30]
    layer: 2 , shape: [None, 32, 32, 30]
    layer: 3 , shape: [None, 64, 64, 3]
    


```python
print('Z_decode shape:', Z_decode.get_shape().as_list())
print('Z_decode_tensor shape:', Z_decode_tensor.get_shape().as_list())            
```

    Z_decode shape: [None, 512]
    Z_decode_tensor shape: [None, 4, 4, 32]
    

Now we need to build our discriminators.  We'll need to add a parameter for the `is_training` placeholder.  We're also going to keep track of every hidden layer in the discriminator.  Our encoder already returns the `Hs` of each layer.  Alternatively, we could poll the graph for each layer in the discriminator and ask for the correspond layer names.  We're going to need these layers when building our costs.


```python
def discriminator(X,
                  is_training,
                  channels=[50, 50, 50, 50],
                  filter_sizes=[4, 4, 4, 4],
                  activation=tf.nn.elu,
                  reuse=None):

    # We'll scope these variables to "discriminator_real"
    with tf.variable_scope('discriminator', reuse=reuse):
        H, Hs = encoder(
            X, is_training, channels, filter_sizes, activation, reuse)
        shape = H.get_shape().as_list()
        print("H: ", H.get_shape().as_list())
        H = tf.reshape(
            H, [-1, shape[1] * shape[2] * shape[3]])
        print("H reshaped: ", H.get_shape().as_list())
        D, W = utils.linear(
            x=H, n_output=1, activation=tf.nn.sigmoid, name='fc', reuse=reuse)
        print("D: ", D)
    return D, Hs
```

Recall the regular GAN and DCGAN required 2 discriminators: one for the generated samples in `Z`, and one for the input samples in `X`.  We'll do the same thing here.  One discriminator for the real input data, `X`, which the discriminator will try to predict as 1s, and another discriminator for the generated samples that go from `X` through the encoder to `Z`, and finally through the decoder to `G`.  The discriminator will be trained to try and predict these as 0s, whereas the generator will be trained to try and predict these as 1s.


```python
D_real, Hs_real = discriminator(X, is_training, reuse=True)
D_fake, Hs_fake = discriminator(G, is_training, reuse=True)
```

    encoder/input: [None, 64, 64, 3]
    layer: 0 , shape: [None, 32, 32, 50]
    layer: 1 , shape: [None, 16, 16, 50]
    layer: 2 , shape: [None, 8, 8, 50]
    layer: 3 , shape: [None, 4, 4, 50]
    H:  [None, 4, 4, 50]
    H reshaped:  [None, 800]
    D:  Tensor("discriminator_4/fc/Sigmoid:0", shape=(?, 1), dtype=float32)
    encoder/input: [None, 64, 64, 3]
    layer: 0 , shape: [None, 32, 32, 50]
    layer: 1 , shape: [None, 16, 16, 50]
    layer: 2 , shape: [None, 8, 8, 50]
    layer: 3 , shape: [None, 4, 4, 50]
    H:  [None, 4, 4, 50]
    H reshaped:  [None, 800]
    D:  Tensor("discriminator_5/fc/Sigmoid:0", shape=(?, 1), dtype=float32)
    

<a name="building-vaegan-loss-functions"></a>
## Building VAE/GAN Loss Functions

Let's now see how we can compose our loss.  We have 3 losses for our discriminator.  Along with measuring the binary cross entropy between each of them, we're going to also measure each layer's loss from our two discriminators using an l2-loss, and this will form our loss for the log likelihood measure.  The details of how these are constructed are explained in more details in the paper: https://arxiv.org/abs/1512.09300 - please refer to this paper for more details that are way beyond the scope of this course!  One parameter within this to pay attention to is `gamma`, which the authors of the paper suggest control the weighting between content and style, just like in Session 4's Style Net implementation.


```python
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
```


```python
nb_utils.show_graph(tf.get_default_graph().as_graph_def())
```



            <iframe seamless style="width:800px;height:620px;border:0" srcdoc="
            <script>
              function load() {
              }
            </script>
            <link rel=&quot;import&quot; href=&quot;https://tensorboard.appspot.com/tf-graph-basic.build.html&quot; onload=load()>
            <div style=&quot;height:600px&quot;>
              <tf-graph-basic id=&quot;graph0.11574998669096859&quot;></tf-graph-basic>
            </div>
        "></iframe>
        


<a name="creating-the-optimizers"></a>
## Creating the Optimizers

We now have losses for our encoder, decoder, and discriminator networks.  We can connect each of these to their own optimizer and start training!  Just like with Part 1's GAN, we'll ensure each network's optimizer only trains its part of the network: the encoder's optimizer will only update the encoder variables, the generator's optimizer will only update the generator variables, and the discriminator's optimizer will only update the discriminator variables.

<h3><font color='red'>TODO! COMPLETE THIS SECTION!</font></h3>


```python
learning_rate = 0.0001

opt_enc = tf.train.AdamOptimizer(
    learning_rate=learning_rate).minimize(
    loss_enc,
    var_list=[var_i for var_i in tf.trainable_variables()
              if var_i.name.startswith('encoder')])

opt_gen = tf.train.AdamOptimizer(
    learning_rate=learning_rate).minimize(
    loss_dec,
    var_list=[var_i for var_i in tf.trainable_variables()
              if var_i.name.startswith('generator')])

opt_dis = tf.train.AdamOptimizer(
    learning_rate=learning_rate).minimize(
    loss_dis,
    var_list=[var_i for var_i in tf.trainable_variables()
              if var_i.name.startswith('discriminator')])
```

<a name="loading-the-dataset"></a>
## Loading the Dataset

We'll now load our dataset just like in Part 1.  Here is where you should explore with your own data!

<h3><font color='red'>TODO! COMPLETE THIS SECTION!</font></h3>


```python
from libs import datasets, dataset_utils

batch_size = 64
n_epochs = 100
crop_shape = [n_pixels, n_pixels, n_channels]
crop_factor = 0.8
input_shape = [218, 178, 3]

# Try w/ CELEB first to make sure it works, then explore w/ your own dataset.
files = datasets.CELEB()
batch = dataset_utils.create_input_pipeline(
    files=files,
    batch_size=batch_size,
    n_epochs=n_epochs,
    crop_shape=crop_shape,
    crop_factor=crop_factor,
    shape=input_shape)
```

    It does not look like you have downloaded the entire Celeb Dataset.
    Try downloading the dataset from the "Aligned and Cropped" link located here (imgs/img_align_celeba.zip [1.34 GB]): http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
    

We'll also create a latent manifold just like we've done in Session 3 and Part 1.  This is a random sampling of 4 points in the latent space of `Z`.  We then interpolate between them to create a "hyper-plane" and show the decoding of 10 x 10 points on that hyperplane.


```python
n_samples = 10
zs = np.random.uniform(
    -1.0, 1.0, [4, n_code]).astype(np.float32)
zs = utils.make_latent_manifold(zs, n_samples)
```

Now create a session and create a coordinator to manage our queues for fetching data from the input pipeline and start our queue runners:


```python
# We create a session to use the graph
sess = tf.Session()
init_op = tf.global_variables_initializer()

saver = tf.train.Saver()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
sess.run(init_op)
```

Load an existing checkpoint if it exists to continue training.


```python
if os.path.exists("vaegan.ckpt"):
    saver.restore(sess, "vaegan.ckpt")
    print("GAN model restored.")
```

We'll also try resynthesizing a test set of images.  This will help us understand how well the encoder/decoder network is doing:


```python
n_files = len(files)
test_xs = sess.run(batch) / 255.0
if not os.path.exists('imgs'):
    os.mkdir('imgs')
m = utils.montage(test_xs, 'imgs/test_xs.png')
plt.imshow(m)
```




    <matplotlib.image.AxesImage at 0x7f88cb635208>




![png](readme_imgs/output_91_1.png)


<a name="training-1"></a>
## Training

Almost ready for training.  Let's get some variables which we'll need.  These are the same as Part 1's training process.  We'll keep track of `t_i` which we'll use to create images of the current manifold and reconstruction every so many iterations.  And we'll keep track of the current batch number within the epoch and the current epoch number.


```python
t_i = 0
batch_i = 0
epoch_i = 0
ckpt_name = './vaegan.ckpt'
```

Just like in Part 1, we'll train trying to maintain an equilibrium between our Generator and Discriminator networks.  You should experiment with the margin depending on how the training proceeds.


```python
equilibrium = 0.693
margin = 0.4
```

Now we'll train!  Just like Part 1, we measure the `real_cost` and `fake_cost`.  But this time, we'll always update the encoder.  Based on the performance of the real/fake costs, then we'll update generator and discriminator networks.  This will take a long time to produce something nice, but not nearly as long as the regular GAN network despite the additional parameters of the encoder and variational networks.  Be sure to monitor the reconstructions to understand when your network has reached the capacity of its learning!  For reference, on Celeb Net, I would use about 5 layers in each of the Encoder, Generator, and Discriminator networks using as input a 100 x 100 image, and a minimum of 200 channels per layer.  This network would take about 1-2 days to train on an Nvidia TITAN X GPU.


```python
while epoch_i < n_epochs:
    if batch_i % (n_files // batch_size) == 0:
        batch_i = 0
        epoch_i += 1
        print('---------- EPOCH:', epoch_i)

    batch_i += 1
    batch_xs = sess.run(batch) / 255.0
    real_cost, fake_cost, _ = sess.run([
        loss_real, loss_fake, opt_enc],
        feed_dict={
            X: batch_xs,
            is_training: True})
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
        sess.run(opt_gen, feed_dict={
            X: batch_xs,
            is_training: True})
    if dis_update:
        sess.run(opt_dis, feed_dict={
            X: batch_xs,
            is_training: True})

    if batch_i % 50 == 0:
        print('real:', real_cost, '/ fake:', fake_cost)

        # Plot example reconstructions from latent layer
        recon = sess.run(G, feed_dict={
                Z: zs,
                is_training: False})

        recon = np.clip(recon, 0, 1)
        m1 = utils.montage(recon.reshape([-1] + crop_shape),
                'imgs/manifold_%08d.png' % t_i)

        # Plot example reconstructions
        recon = sess.run(G, feed_dict={
                X: test_xs,
                is_training: False})
        recon = np.clip(recon, 0, 1)
        m2 = utils.montage(recon.reshape([-1] + crop_shape),
                'imgs/reconstruction_%08d.png' % t_i)
        
        fig, axs = plt.subplots(1, 2, figsize=(15, 10))
        axs[0].imshow(m1)
        axs[1].imshow(m2)
        plt.show()
        
        t_i += 1
        

    if batch_i % 200 == 0:
        # Save the variables to disk.
        save_path = saver.save(sess, "./" + ckpt_name,
                               global_step=batch_i,
                               write_meta_graph=False)
        print("Model saved in file: %s" % save_path)
```

    ---------- EPOCH: 1
    real: 0.205149 / fake: 0.971561
    


![png](readme_imgs/output_97_1.png)


    real: 0.148093 / fake: 0.984728
    


![png](readme_imgs/output_97_3.png)


    real: 0.172682 / fake: 0.895907
    


![png](readme_imgs/output_97_5.png)


    real: 0.206726 / fake: 1.18948
    


![png](readme_imgs/output_97_7.png)


    Model saved in file: ././vaegan.ckpt-200
    real: 0.473749 / fake: 0.203739
    


![png](readme_imgs/output_97_9.png)


    real: 0.430192 / fake: 0.225813
    


![png](readme_imgs/output_97_11.png)


    real: 0.235843 / fake: 0.337034
    


![png](readme_imgs/output_97_13.png)


    real: 0.225244 / fake: 0.450669
    


![png](readme_imgs/output_97_15.png)


    Model saved in file: ././vaegan.ckpt-400
    real: 0.236165 / fake: 0.628434
    


![png](readme_imgs/output_97_17.png)


    real: 0.282716 / fake: 0.741345
    


![png](readme_imgs/output_97_19.png)


    real: 0.210514 / fake: 0.93315
    


![png](readme_imgs/output_97_21.png)


    real: 0.223767 / fake: 1.06233
    


![png](readme_imgs/output_97_23.png)


    Model saved in file: ././vaegan.ckpt-600
    real: 0.296179 / fake: 0.532847
    


![png](readme_imgs/output_97_25.png)


    real: 0.54699 / fake: 0.172794
    


![png](readme_imgs/output_97_27.png)


    real: 0.458605 / fake: 0.201232
    


![png](readme_imgs/output_97_29.png)


    real: 0.455976 / fake: 0.25526
    


![png](readme_imgs/output_97_31.png)


    Model saved in file: ././vaegan.ckpt-800
    real: 0.525117 / fake: 0.218225
    


![png](readme_imgs/output_97_33.png)


    real: 0.508663 / fake: 0.210913
    


![png](readme_imgs/output_97_35.png)


    real: 0.478644 / fake: 0.177155
    


![png](readme_imgs/output_97_37.png)


    real: 0.526226 / fake: 0.273313
    


![png](readme_imgs/output_97_39.png)


    Model saved in file: ././vaegan.ckpt-1000
    real: 0.53386 / fake: 0.221569
    


![png](readme_imgs/output_97_41.png)


    real: 0.442767 / fake: 0.243084
    


![png](readme_imgs/output_97_43.png)


    real: 0.420646 / fake: 0.212092
    


![png](readme_imgs/output_97_45.png)


    real: 0.377074 / fake: 0.267553
    


![png](readme_imgs/output_97_47.png)


    Model saved in file: ././vaegan.ckpt-1200
    real: 0.322817 / fake: 0.297871
    


![png](readme_imgs/output_97_49.png)


    real: 0.26702 / fake: 0.259011
    


![png](readme_imgs/output_97_51.png)


    real: 0.286308 / fake: 0.242103
    


![png](readme_imgs/output_97_53.png)


    real: 0.230102 / fake: 0.25184
    


![png](readme_imgs/output_97_55.png)


    Model saved in file: ././vaegan.ckpt-1400
    real: 0.308311 / fake: 0.316554
    


![png](readme_imgs/output_97_57.png)


    real: 0.297514 / fake: 0.244885
    


![png](readme_imgs/output_97_59.png)


    real: 0.250035 / fake: 0.259994
    


![png](readme_imgs/output_97_61.png)


    real: 0.333895 / fake: 0.243069
    


![png](readme_imgs/output_97_63.png)


    Model saved in file: ././vaegan.ckpt-1600
    real: 0.271636 / fake: 0.234075
    


![png](readme_imgs/output_97_65.png)


    real: 0.289715 / fake: 0.273397
    


![png](readme_imgs/output_97_67.png)


    real: 0.259443 / fake: 0.326204
    


![png](readme_imgs/output_97_69.png)


    real: 0.266879 / fake: 0.281439
    


![png](readme_imgs/output_97_71.png)


    Model saved in file: ././vaegan.ckpt-1800
    real: 0.27175 / fake: 0.193356
    


![png](readme_imgs/output_97_73.png)


    real: 0.28646 / fake: 0.282162
    


![png](readme_imgs/output_97_75.png)


    real: 0.302956 / fake: 0.19802
    


![png](readme_imgs/output_97_77.png)


    real: 0.362498 / fake: 0.222033
    


![png](readme_imgs/output_97_79.png)


    Model saved in file: ././vaegan.ckpt-2000
    real: 0.341877 / fake: 0.229757
    


```python
# One of the threads has issued an exception.  So let's tell all the
# threads to shutdown.
coord.request_stop()

# Wait until all threads have finished.
coord.join(threads)

# Clean up the session.
sess.close()
```

<a name="part-3---latent-space-arithmetic"></a>
# Part 3 - Latent-Space Arithmetic

<a name="loading-the-pre-trained-model"></a>
## Loading the Pre-Trained Model

We're now going to work with a pre-trained VAEGAN model on the Celeb Net dataset.  Let's load this model:


```python
tf.reset_default_graph()
```


```python
from libs import celeb_vaegan as CV
```


```python
net = CV.get_celeb_vaegan_model()
```

We'll load the graph_def contained inside this dictionary.  It follows the same idea as the `inception`, `vgg16`, and `i2v` pretrained networks.  It is a dictionary with the key `graph_def` defined, with the graph's pretrained network.  It also includes `labels` and a `preprocess` key.  We'll have to do one additional thing which is to turn off the random sampling from variational layer.  This isn't really necessary but will ensure we get the same results each time we use the network.  We'll use the `input_map` argument to do this.  Don't worry if this doesn't make any sense, as we didn't cover the variational layer in any depth.  Just know that this is removing a random process from the network so that it is completely deterministic.  If we hadn't done this, we'd get slightly different results each time we used the network (which may even be desirable for your purposes).


```python
sess = tf.Session()
g = tf.get_default_graph()
tf.import_graph_def(net['graph_def'], name='net', input_map={
        'encoder/variational/random_normal:0': np.zeros(512, dtype=np.float32)})
names = [op.name for op in g.get_operations()]
print(names)
```

    ['net/_inputs/Const', 'net/x', 'net/encoder/0/conv2d/W', 'net/encoder/0/conv2d/W/read', 'net/encoder/0/conv2d/conv', 'net/encoder/0/conv2d/b', 'net/encoder/0/conv2d/b/read', 'net/encoder/0/conv2d/h', 'net/encoder/0/Elu', 'net/encoder/1/conv2d/W', 'net/encoder/1/conv2d/W/read', 'net/encoder/1/conv2d/conv', 'net/encoder/1/conv2d/b', 'net/encoder/1/conv2d/b/read', 'net/encoder/1/conv2d/h', 'net/encoder/1/Elu', 'net/encoder/2/conv2d/W', 'net/encoder/2/conv2d/W/read', 'net/encoder/2/conv2d/conv', 'net/encoder/2/conv2d/b', 'net/encoder/2/conv2d/b/read', 'net/encoder/2/conv2d/h', 'net/encoder/2/Elu', 'net/encoder/3/conv2d/W', 'net/encoder/3/conv2d/W/read', 'net/encoder/3/conv2d/conv', 'net/encoder/3/conv2d/b', 'net/encoder/3/conv2d/b/read', 'net/encoder/3/conv2d/h', 'net/encoder/3/Elu', 'net/encoder/4/conv2d/W', 'net/encoder/4/conv2d/W/read', 'net/encoder/4/conv2d/conv', 'net/encoder/4/conv2d/b', 'net/encoder/4/conv2d/b/read', 'net/encoder/4/conv2d/h', 'net/encoder/4/Elu', 'net/encoder/flatten/flatten/Reshape/shape', 'net/encoder/flatten/flatten/Reshape', 'net/encoder/variational/mu/W', 'net/encoder/variational/mu/W/read', 'net/encoder/variational/mu/b', 'net/encoder/variational/mu/b/read', 'net/encoder/variational/mu/MatMul', 'net/encoder/variational/mu/h', 'net/encoder/variational/Tanh', 'net/encoder/variational/log_sigma/W', 'net/encoder/variational/log_sigma/W/read', 'net/encoder/variational/log_sigma/b', 'net/encoder/variational/log_sigma/b/read', 'net/encoder/variational/log_sigma/MatMul', 'net/encoder/variational/log_sigma/h', 'net/encoder/variational/Tanh_1', 'net/encoder/variational/mul/x', 'net/encoder/variational/mul', 'net/encoder/variational/Shape', 'net/encoder/variational/Slice/begin', 'net/encoder/variational/Slice/size', 'net/encoder/variational/Slice', 'net/encoder/variational/Squeeze', 'net/encoder/variational/pack/1', 'net/encoder/variational/pack', 'net/encoder/variational/random_normal/mean', 'net/encoder/variational/random_normal/stddev', 'net/encoder/variational/random_normal/RandomStandardNormal', 'net/encoder/variational/random_normal/mul', 'net/encoder/variational/random_normal', 'net/encoder/variational/Exp', 'net/encoder/variational/Mul', 'net/encoder/variational/z', 'net/generator/hidden/2/linear/W', 'net/generator/hidden/2/linear/W/read', 'net/generator/hidden/2/linear/b', 'net/generator/hidden/2/linear/b/read', 'net/generator/hidden/2/linear/MatMul', 'net/generator/hidden/2/linear/h', 'net/generator/hidden/2/Elu', 'net/generator/hidden/2/Shape', 'net/generator/hidden/2/Slice/begin', 'net/generator/hidden/2/Slice/size', 'net/generator/hidden/2/Slice', 'net/generator/hidden/2/Squeeze', 'net/generator/hidden/2/pack/1', 'net/generator/hidden/2/pack/2', 'net/generator/hidden/2/pack/3', 'net/generator/hidden/2/pack', 'net/generator/hidden/2/Reshape', 'net/generator/decoder/0/deconv2d/W', 'net/generator/decoder/0/deconv2d/W/read', 'net/generator/decoder/0/deconv2d/Shape', 'net/generator/decoder/0/deconv2d/Slice/begin', 'net/generator/decoder/0/deconv2d/Slice/size', 'net/generator/decoder/0/deconv2d/Slice', 'net/generator/decoder/0/deconv2d/Squeeze', 'net/generator/decoder/0/deconv2d/pack/1', 'net/generator/decoder/0/deconv2d/pack/2', 'net/generator/decoder/0/deconv2d/pack/3', 'net/generator/decoder/0/deconv2d/pack', 'net/generator/decoder/0/deconv2d/conv_t', 'net/generator/decoder/0/deconv2d/b', 'net/generator/decoder/0/deconv2d/b/read', 'net/generator/decoder/0/deconv2d/h', 'net/generator/decoder/0/Elu', 'net/generator/decoder/1/deconv2d/W', 'net/generator/decoder/1/deconv2d/W/read', 'net/generator/decoder/1/deconv2d/Shape', 'net/generator/decoder/1/deconv2d/Slice/begin', 'net/generator/decoder/1/deconv2d/Slice/size', 'net/generator/decoder/1/deconv2d/Slice', 'net/generator/decoder/1/deconv2d/Squeeze', 'net/generator/decoder/1/deconv2d/pack/1', 'net/generator/decoder/1/deconv2d/pack/2', 'net/generator/decoder/1/deconv2d/pack/3', 'net/generator/decoder/1/deconv2d/pack', 'net/generator/decoder/1/deconv2d/conv_t', 'net/generator/decoder/1/deconv2d/b', 'net/generator/decoder/1/deconv2d/b/read', 'net/generator/decoder/1/deconv2d/h', 'net/generator/decoder/1/Elu', 'net/generator/decoder/2/deconv2d/W', 'net/generator/decoder/2/deconv2d/W/read', 'net/generator/decoder/2/deconv2d/Shape', 'net/generator/decoder/2/deconv2d/Slice/begin', 'net/generator/decoder/2/deconv2d/Slice/size', 'net/generator/decoder/2/deconv2d/Slice', 'net/generator/decoder/2/deconv2d/Squeeze', 'net/generator/decoder/2/deconv2d/pack/1', 'net/generator/decoder/2/deconv2d/pack/2', 'net/generator/decoder/2/deconv2d/pack/3', 'net/generator/decoder/2/deconv2d/pack', 'net/generator/decoder/2/deconv2d/conv_t', 'net/generator/decoder/2/deconv2d/b', 'net/generator/decoder/2/deconv2d/b/read', 'net/generator/decoder/2/deconv2d/h', 'net/generator/decoder/2/Elu', 'net/generator/decoder/3/deconv2d/W', 'net/generator/decoder/3/deconv2d/W/read', 'net/generator/decoder/3/deconv2d/Shape', 'net/generator/decoder/3/deconv2d/Slice/begin', 'net/generator/decoder/3/deconv2d/Slice/size', 'net/generator/decoder/3/deconv2d/Slice', 'net/generator/decoder/3/deconv2d/Squeeze', 'net/generator/decoder/3/deconv2d/pack/1', 'net/generator/decoder/3/deconv2d/pack/2', 'net/generator/decoder/3/deconv2d/pack/3', 'net/generator/decoder/3/deconv2d/pack', 'net/generator/decoder/3/deconv2d/conv_t', 'net/generator/decoder/3/deconv2d/b', 'net/generator/decoder/3/deconv2d/b/read', 'net/generator/decoder/3/deconv2d/h', 'net/generator/decoder/3/Elu', 'net/generator/decoder/4/deconv2d/W', 'net/generator/decoder/4/deconv2d/W/read', 'net/generator/decoder/4/deconv2d/Shape', 'net/generator/decoder/4/deconv2d/Slice/begin', 'net/generator/decoder/4/deconv2d/Slice/size', 'net/generator/decoder/4/deconv2d/Slice', 'net/generator/decoder/4/deconv2d/Squeeze', 'net/generator/decoder/4/deconv2d/pack/1', 'net/generator/decoder/4/deconv2d/pack/2', 'net/generator/decoder/4/deconv2d/pack/3', 'net/generator/decoder/4/deconv2d/pack', 'net/generator/decoder/4/deconv2d/conv_t', 'net/generator/decoder/4/deconv2d/b', 'net/generator/decoder/4/deconv2d/b/read', 'net/generator/decoder/4/deconv2d/h', 'net/generator/decoder/4/Elu', 'net/generator/x_tilde']
    

Now let's get the relevant parts of the network: `X`, the input image to the network, `Z`, the input image's encoding, and `G`, the decoded image.  In many ways, this is just like the Autoencoders we learned about in Session 3, except instead of `Y` being the output, we have `G` from our generator!  And the way we train it is very different: we use an adversarial process between the generator and discriminator, and use the discriminator's own distance measure to help train the network, rather than pixel-to-pixel differences.


```python
X = g.get_tensor_by_name('net/x:0')
Z = g.get_tensor_by_name('net/encoder/variational/z:0')
G = g.get_tensor_by_name('net/generator/x_tilde:0')
```

Let's get some data to play with:


```python
files = datasets.CELEB()
img_i = 50
img = plt.imread(files[img_i])
plt.imshow(img)
```




    <matplotlib.image.AxesImage at 0x15c269ca8d0>




![png](readme_imgs/output_108_1.png)


Now preprocess the image, and see what the generated image looks like (i.e. the lossy version of the image through the network's encoding and decoding).


```python
p = CV.preprocess(img)
synth = sess.run(G, feed_dict={X: p[np.newaxis]})
```

    C:\Users\eutan\Anaconda3\lib\site-packages\skimage\transform\_warps.py:84: UserWarning: The default mode, 'constant', will be changed to 'reflect' in skimage 0.15.
      warn("The default mode, 'constant', will be changed to 'reflect' in "
    


```python
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(p)
axs[1].imshow(synth[0] / synth.max())
```




    <matplotlib.image.AxesImage at 0x15c26ac2ac8>




![png](readme_imgs/output_111_1.png)


So we lost a lot of details but it seems to be able to express quite a bit about the image.  Our inner most layer, `Z`, is only 512 values yet our dataset was 200k images of 64 x 64 x 3 pixels (about 2.3 GB of information).  That means we're able to express our nearly 2.3 GB of information with only 512 values!  Having some loss of detail is certainly expected!

<a name="exploring-the-celeb-net-attributes"></a>
## Exploring the Celeb Net Attributes

Let's now try and explore the attributes of our dataset.  We didn't train the network with any supervised labels, but the Celeb Net dataset has 40 attributes for each of its 200k images.  These are already parsed and stored for you in the `net` dictionary:


```python
net.keys()
```




    dict_keys(['graph_def', 'labels', 'attributes', 'preprocess'])




```python
len(net['labels'])
```




    40




```python
net['labels']
```




    ['5_o_Clock_Shadow',
     'Arched_Eyebrows',
     'Attractive',
     'Bags_Under_Eyes',
     'Bald',
     'Bangs',
     'Big_Lips',
     'Big_Nose',
     'Black_Hair',
     'Blond_Hair',
     'Blurry',
     'Brown_Hair',
     'Bushy_Eyebrows',
     'Chubby',
     'Double_Chin',
     'Eyeglasses',
     'Goatee',
     'Gray_Hair',
     'Heavy_Makeup',
     'High_Cheekbones',
     'Male',
     'Mouth_Slightly_Open',
     'Mustache',
     'Narrow_Eyes',
     'No_Beard',
     'Oval_Face',
     'Pale_Skin',
     'Pointy_Nose',
     'Receding_Hairline',
     'Rosy_Cheeks',
     'Sideburns',
     'Smiling',
     'Straight_Hair',
     'Wavy_Hair',
     'Wearing_Earrings',
     'Wearing_Hat',
     'Wearing_Lipstick',
     'Wearing_Necklace',
     'Wearing_Necktie',
     'Young']



Let's see what attributes exist for one of the celeb images:


```python
plt.imshow(img)
[net['labels'][i] for i, attr_i in enumerate(net['attributes'][img_i]) if attr_i]
```




    ['5_o_Clock_Shadow',
     'Bags_Under_Eyes',
     'Bald',
     'Big_Nose',
     'Chubby',
     'Double_Chin',
     'Gray_Hair',
     'Male',
     'Receding_Hairline']




![png](readme_imgs/output_117_1.png)


<a name="find-the-latent-encoding-for-an-attribute"></a>
## Find the Latent Encoding for an Attribute

The Celeb Dataset includes attributes for each of its 200k+ images.  This allows us to feed into the encoder some images that we know have a *specific* attribute, e.g. "smiling".  We store what their encoding is and retain this distribution of encoded values.  We can then look at any other image and see how it is encoded, and slightly change the encoding by adding the encoded of our smiling images to it!  The result should be our image but with more smiling.  That is just insane and we're going to see how to do it.  First lets inspect our latent space:


```python
Z.get_shape()
```




    TensorShape([Dimension(None), Dimension(512)])



We have 512 features that we can encode any image with.  Assuming our network is doing an okay job, let's try to find the `Z` of the first 100 images with the 'Bald' attribute:


```python
bald_label = net['labels'].index('Bald')
```


```python
bald_label
```




    4



Let's get all the bald image indexes:


```python
bald_img_idxs = np.where(net['attributes'][:, bald_label])[0]
```


```python
bald_img_idxs
```




    array([    50,     78,    114, ..., 202503, 202524, 202560])



Now let's just load 100 of their images:


```python
bald_imgs = [plt.imread(files[bald_img_i])[..., :3]
             for bald_img_i in bald_img_idxs[:100]]
```

Let's see if the mean image looks like a good bald person or not:


```python
plt.imshow(np.mean(bald_imgs, 0).astype(np.uint8))
```




    <matplotlib.image.AxesImage at 0x7f757e0c1710>




![png](readme_imgs/output_129_1.png)


Yes that is definitely a bald person.  Now we're going to try to find the encoding of a bald person.  One method is to try and find every other possible image and subtract the "bald" person's latent encoding.  Then we could add this encoding back to any new image and hopefully it makes the image look more bald.  Or we can find a bunch of bald people's encodings and then average their encodings together.  This should reduce the noise from having many different attributes, but keep the signal pertaining to the baldness.

Let's first preprocess the images:


```python
bald_p = np.array([CV.preprocess(bald_img_i) for bald_img_i in bald_imgs])
bald_p.shape
```

    /home/ubuntu/.local/lib/python3.5/site-packages/skimage/transform/_warps.py:84: UserWarning: The default mode, 'constant', will be changed to 'reflect' in skimage 0.15.
      warn("The default mode, 'constant', will be changed to 'reflect' in "
    




    (100, 100, 100, 3)



Now we can find the latent encoding of the images by calculating `Z` and feeding `X` with our `bald_p` images:

<h3><font color='red'>TODO! COMPLETE THIS SECTION!</font></h3>


```python
bald_zs = sess.run(Z, feed_dict={X: bald_p})
```

Now let's calculate the mean encoding:


```python
bald_feature = np.mean(bald_zs, 0, keepdims=True)
```


```python
bald_feature.shape
```




    (1, 512)



Let's try and synthesize from the mean bald feature now and see how it looks:

<h3><font color='red'>TODO! COMPLETE THIS SECTION!</font></h3>


```python
bald_generated = sess.run(G, feed_dict={Z: bald_feature})
```


```python
plt.imshow(bald_generated[0] / bald_generated.max())
```




    <matplotlib.image.AxesImage at 0x7fae3763f048>




![png](readme_imgs/output_139_1.png)


<a name="latent-feature-arithmetic"></a>
## Latent Feature Arithmetic

Let's now try to write a general function for performing everything we've just done so that we can do this with many different features.  We'll then try to combine them and synthesize people with the features we want them to have...


```python
def get_features_for(label='Bald', has_label=True, n_imgs=50):
    label_i = net['labels'].index(label)
    label_idxs = np.where(net['attributes'][:, label_i] == has_label)[0]
    label_idxs = np.random.permutation(label_idxs)[:n_imgs]
    imgs = [plt.imread(files[img_i])[..., :3]
            for img_i in label_idxs]
    preprocessed = np.array([CV.preprocess(img_i) for img_i in imgs])
    zs = sess.run(Z, feed_dict={X: preprocessed})
    return np.mean(zs, 0)
```

Let's try getting some attributes positive and negative features.  Be sure to explore different attributes!  Also try different values of `n_imgs`, e.g. 2, 3, 5, 10, 50, 100.  What happens with different values?

<h3><font color='red'>TODO! COMPLETE THIS SECTION!</font></h3>


```python
# Explore different attributes
z1 = get_features_for('Male', True, n_imgs=10)
z2 = get_features_for('Male', False, n_imgs=10)
z3 = get_features_for('Smiling', True, n_imgs=10)
z4 = get_features_for('Smiling', False, n_imgs=10)
```

    /home/ubuntu/.local/lib/python3.5/site-packages/skimage/transform/_warps.py:84: UserWarning: The default mode, 'constant', will be changed to 'reflect' in skimage 0.15.
      warn("The default mode, 'constant', will be changed to 'reflect' in "
    


```python
b1 = sess.run(G, feed_dict={Z: z1[np.newaxis]})
b2 = sess.run(G, feed_dict={Z: z2[np.newaxis]})
b3 = sess.run(G, feed_dict={Z: z3[np.newaxis]})
b4 = sess.run(G, feed_dict={Z: z4[np.newaxis]})
```


```python
fig, axs = plt.subplots(1, 4, figsize=(15, 6))
axs[0].imshow(b1[0] / b1.max()), axs[0].set_title('Male'), axs[0].grid('off'), axs[0].axis('off')
axs[1].imshow(b2[0] / b2.max()), axs[1].set_title('Not Male'), axs[1].grid('off'), axs[1].axis('off')
axs[2].imshow(b3[0] / b3.max()), axs[2].set_title('Smiling'), axs[2].grid('off'), axs[2].axis('off')
axs[3].imshow(b4[0] / b4.max()), axs[3].set_title('Not Smiling'), axs[3].grid('off'), axs[3].axis('off')
```




    (<matplotlib.image.AxesImage at 0x7f948ca6e3c8>,
     <matplotlib.text.Text at 0x7f948ca4aa20>,
     None,
     (-0.5, 99.5, 99.5, -0.5))




![png](readme_imgs/output_145_1.png)


Now let's interpolate between the "Male" and "Not Male" categories:


```python
notmale_vector = z2 - z1
n_imgs = 5
amt = np.linspace(0, 1, n_imgs)
zs = np.array([z1 + notmale_vector*amt_i for amt_i in amt])
g = sess.run(G, feed_dict={Z: zs})
```


```python
fig, axs = plt.subplots(1, n_imgs, figsize=(20, 4))
for i, ax_i in enumerate(axs):
    ax_i.imshow(np.clip(g[i], 0, 1))
    ax_i.grid('off')
    ax_i.axis('off')
```


![png](readme_imgs/output_148_0.png)


And the same for smiling:


```python
smiling_vector = z3 - z4
amt = np.linspace(0, 1, n_imgs)
zs = np.array([z4 + smiling_vector*amt_i for amt_i in amt])
g = sess.run(G, feed_dict={Z: zs})
fig, axs = plt.subplots(1, n_imgs, figsize=(20, 4))
for i, ax_i in enumerate(axs):
    ax_i.imshow(np.clip(g[i] / g[i].max(), 0, 1))
    ax_i.grid('off')
```


![png](readme_imgs/output_150_0.png)


There's also no reason why we have to be within the boundaries of 0-1.  We can extrapolate beyond, in, and around the space.


```python
n_imgs = 5
amt = np.linspace(-1.5, 2.5, n_imgs)
zs = np.array([z4 + smiling_vector*amt_i for amt_i in amt])
g = sess.run(G, feed_dict={Z: zs})
fig, axs = plt.subplots(1, n_imgs, figsize=(20, 4))
for i, ax_i in enumerate(axs):
    ax_i.imshow(np.clip(g[i], 0, 1))
    ax_i.grid('off')
    ax_i.axis('off')
```


![png](readme_imgs/output_152_0.png)


<a name="extensions"></a>
## Extensions

[Tom White](https://twitter.com/dribnet), Lecturer at Victoria University School of Design, also recently demonstrated an alternative way of interpolating using a sinusoidal interpolation.  He's created some of the most impressive generative images out there and luckily for us he has detailed his process in the arxiv preprint: https://arxiv.org/abs/1609.04468 - as well, be sure to check out his twitter bot, https://twitter.com/smilevector - which adds smiles to people :) - Note that the network we're using is only trained on aligned faces that are frontally facing, though this twitter bot is capable of adding smiles to any face.  I suspect that he is running a face detection algorithm such as AAM, CLM, or ASM, cropping the face, aligning it, and then running a similar algorithm to what we've done above.  Or else, perhaps he has trained a new model on faces that are not aligned.  In any case, it is well worth checking out!

Let's now try and use sinusoidal interpolation using his implementation in [plat](https://github.com/dribnet/plat/blob/master/plat/interpolate.py#L16-L24) which I've copied below:


```python
def slerp(val, low, high):
    """Spherical interpolation. val has a range of 0 to 1."""
    if val <= 0:
        return low
    elif val >= 1:
        return high
    omega = np.arccos(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega)/so * high
```


```python
amt = np.linspace(0, 1, n_imgs)
zs = np.array([slerp(amt_i, z1, z2) for amt_i in amt])
g = sess.run(G, feed_dict={Z: zs})
fig, axs = plt.subplots(1, n_imgs, figsize=(20, 4))
for i, ax_i in enumerate(axs):
    ax_i.imshow(np.clip(g[i], 0, 1))
    ax_i.grid('off')
    ax_i.axis('off')
```


![png](readme_imgs/output_155_0.png)


It's certainly worth trying especially if you are looking to explore your own model's latent space in new and interesting ways.

Let's try and load an image that we want to play with.  We need an image as similar to the Celeb Dataset as possible.  Unfortunately, we don't have access to the algorithm they used to "align" the faces, so we'll need to try and get as close as possible to an aligned face image.  One way you can do this is to load up one of the celeb images and try and align an image to it using e.g. Photoshop or another photo editing software that lets you blend and move the images around.  That's what I did for my own face...


```python
img = plt.imread('parag.png')[..., :3]
img = CV.preprocess(img, crop_factor=1.0)[np.newaxis]
```

    /home/ubuntu/.local/lib/python3.5/site-packages/skimage/transform/_warps.py:84: UserWarning: The default mode, 'constant', will be changed to 'reflect' in skimage 0.15.
      warn("The default mode, 'constant', will be changed to 'reflect' in "
    

Let's see how the network encodes it:


```python
img_ = sess.run(G, feed_dict={X: img})
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(img[0]), axs[0].grid('off')
axs[1].imshow(np.clip(img_[0] / np.max(img_), 0, 1)), axs[1].grid('off')
```




    (<matplotlib.image.AxesImage at 0x7f948baf7390>, None)




![png](readme_imgs/output_159_1.png)


Notice how blurry the image is.  Tom White's preprint suggests one way to sharpen the image is to find the "Blurry" attribute vector:


```python
z1 = get_features_for('Blurry', True, n_imgs=25)
z2 = get_features_for('Blurry', False, n_imgs=25)
unblur_vector = z2 - z1
```

    /home/ubuntu/.local/lib/python3.5/site-packages/skimage/transform/_warps.py:84: UserWarning: The default mode, 'constant', will be changed to 'reflect' in skimage 0.15.
      warn("The default mode, 'constant', will be changed to 'reflect' in "
    


```python
z = sess.run(Z, feed_dict={X: img})
```


```python
n_imgs = 5
amt = np.linspace(0, 1, n_imgs)
zs = np.array([z[0] + unblur_vector * amt_i for amt_i in amt])
g = sess.run(G, feed_dict={Z: zs})
fig, axs = plt.subplots(1, n_imgs, figsize=(20, 4))
for i, ax_i in enumerate(axs):
    ax_i.imshow(np.clip(g[i] / g[i].max(), 0, 1))
    ax_i.grid('off')
    ax_i.axis('off')
```


![png](readme_imgs/output_163_0.png)


Notice that the image also gets brighter and perhaps other features than simply the bluriness of the image changes.  Tom's preprint suggests that this is due to the correlation that blurred images have with other things such as the brightness of the image, possibly due biases in labeling or how photographs are taken.   He suggests that another way to unblur would be to synthetically blur a set of images and find the difference in the encoding between the real and blurred images.  We can try it like so:


```python
from scipy.ndimage import gaussian_filter

idxs = np.random.permutation(range(len(files)))
imgs = [plt.imread(files[idx_i]) for idx_i in idxs[:100]]
blurred = []
for img_i in imgs:
    img_copy = np.zeros_like(img_i)
    for ch_i in range(3):
        img_copy[..., ch_i] = gaussian_filter(img_i[..., ch_i], sigma=3.0)
    blurred.append(img_copy)
```


```python
# Now let's preprocess the original images and the blurred ones
imgs_p = np.array([CV.preprocess(img_i) for img_i in imgs])
blur_p = np.array([CV.preprocess(img_i) for img_i in blurred])

# And then compute each of their latent features
noblur = sess.run(Z, feed_dict={X: imgs_p})
blur = sess.run(Z, feed_dict={X: blur_p})
```

    /home/ubuntu/.local/lib/python3.5/site-packages/skimage/transform/_warps.py:84: UserWarning: The default mode, 'constant', will be changed to 'reflect' in skimage 0.15.
      warn("The default mode, 'constant', will be changed to 'reflect' in "
    


```python
synthetic_unblur_vector = np.mean(noblur - blur, 0)
```


```python
n_imgs = 5
amt = np.linspace(0, 1, n_imgs)
zs = np.array([z[0] + synthetic_unblur_vector * amt_i for amt_i in amt])
g = sess.run(G, feed_dict={Z: zs})
fig, axs = plt.subplots(1, n_imgs, figsize=(20, 4))
for i, ax_i in enumerate(axs):
    ax_i.imshow(np.clip(g[i], 0, 1))
    ax_i.grid('off')
    ax_i.axis('off')
```


![png](readme_imgs/output_168_0.png)


For some reason, it also doesn't like my glasses very much.  Let's try and add them back.


```python
z1 = get_features_for('Eyeglasses', True)
z2 = get_features_for('Eyeglasses', False)
glass_vector = z1 - z2
```

    /home/ubuntu/.local/lib/python3.5/site-packages/skimage/transform/_warps.py:84: UserWarning: The default mode, 'constant', will be changed to 'reflect' in skimage 0.15.
      warn("The default mode, 'constant', will be changed to 'reflect' in "
    


```python
z = sess.run(Z, feed_dict={X: img})
```


```python
n_imgs = 5
amt = np.linspace(0, 1, n_imgs)
zs = np.array([z[0] + glass_vector * amt_i + unblur_vector * amt_i for amt_i in amt])
g = sess.run(G, feed_dict={Z: zs})
fig, axs = plt.subplots(1, n_imgs, figsize=(20, 4))
for i, ax_i in enumerate(axs):
    ax_i.imshow(np.clip(g[i], 0, 1))
    ax_i.grid('off')
    ax_i.axis('off')
```


![png](readme_imgs/output_172_0.png)


Well, more like sunglasses then.  Let's try adding everything in there now!


```python
n_imgs = 5
amt = np.linspace(0, 1.0, n_imgs)
zs = np.array([z[0] + glass_vector * amt_i + unblur_vector * amt_i + amt_i * smiling_vector for amt_i in amt])
g = sess.run(G, feed_dict={Z: zs})
fig, axs = plt.subplots(1, n_imgs, figsize=(20, 4))
for i, ax_i in enumerate(axs):
    ax_i.imshow(np.clip(g[i], 0, 1))
    ax_i.grid('off')
    ax_i.axis('off')
```


![png](readme_imgs/output_174_0.png)


Well it was worth a try anyway.  We can also try with a lot of images and create a gif montage of the result:


```python
n_imgs = 5
amt = np.linspace(0, 1.5, n_imgs)
z = sess.run(Z, feed_dict={X: imgs_p})
imgs = []
for amt_i in amt:
    zs = z + synthetic_unblur_vector * amt_i + amt_i * smiling_vector
    g = sess.run(G, feed_dict={Z: zs})
    m = utils.montage(np.clip(g, 0, 1))
    imgs.append(m)
```


```python
import imageio
imageio.mimsave('celeb.gif', imgs)
```

    /home/ubuntu/.local/lib/python3.5/site-packages/imageio/core/util.py:78: UserWarning: Lossy conversion from float64 to uint8, range [0, 1]
      dtype_str, out_type.__name__))
    


```python
ipyd.Image(url='celeb.gif?i={}'.format(
        np.random.rand()), height=1000, width=1000)
```




<img src="celeb.gif?i=0.7070193762216292" width="1000" height="1000"/>



Exploring multiple feature vectors and applying them to images from the celeb dataset to produce animations of a face, saving it as a GIF.  Recall you can store each image frame in a list and then use the `gif.build_gif` function to create a gif.  Explore your own syntheses and then include a gif of the different images you create as "celeb.gif" in the final submission.  Perhaps try finding unexpected synthetic latent attributes in the same way that we created a blur attribute.  You can check the documentation in scipy.ndimage for some other image processing techniques, for instance: http://www.scipy-lectures.org/advanced/image_processing/ - and see if you can find the encoding of another attribute that you then apply to your own images.  You can even try it with many images and use the `utils.montage` function to create a large grid of images that evolves over your attributes.  Or create a set of expressions perhaps.  Up to you just explore!




<h3><font color='red'>TODO! COMPLETE THIS SECTION!</font></h3>

The objective is to transition unsmiling images to Young, Attractive and Smiling images, followed by Grey, Chubby, Unsmiling with Eye Bags


```python
# Obtain first 100 unsmiling images
label_unsmile_i = net['labels'].index('Smiling')
label_unsmile_idxs = np.where(net['attributes'][:, label_unsmile_i] == False)[0]
label_unsmile_idxs = np.random.permutation(label_unsmile_idxs)[:100]
imgs_unsmile = [plt.imread(files[img_i])[..., :3]
        for img_i in label_unsmile_idxs]

x = np.array(imgs_unsmile)
x = np.array([CV.preprocess(img_i) for img_i in imgs_unsmile])

# Unblur the images
idxs = np.random.permutation(range(len(files)))
imgs = imgs_unsmile
blurred = []
for img_i in imgs:
    img_copy = np.zeros_like(img_i)
    for ch_i in range(3):
        img_copy[..., ch_i] = gaussian_filter(img_i[..., ch_i], sigma=3.0)
    blurred.append(img_copy)
# Now let's preprocess the original images and the blurred ones
imgs_p = np.array([CV.preprocess(img_i) for img_i in imgs])
blur_p = np.array([CV.preprocess(img_i) for img_i in blurred])

# And then compute each of their latent features
noblur = sess.run(Z, feed_dict={X: imgs_p})
blur = sess.run(Z, feed_dict={X: blur_p})
synthetic_unblur_vector = np.mean(noblur - blur, 0)

# Smiling features
z_smile = get_features_for(label='Smiling', has_label=True, n_imgs=50)
z_unsmile = get_features_for(label='Smiling', has_label=False, n_imgs=50)

# Attractive features
z_attractive = get_features_for(label='Attractive', has_label=True, n_imgs=50)
z_unattractive = get_features_for(label='Attractive', has_label=False, n_imgs=50)

# Young features
z_young = get_features_for(label='Young', has_label=True, n_imgs=50)
z_unyoung = get_features_for(label='Young', has_label=False, n_imgs=50)

# Eye Bags features
z_bag = get_features_for(label='Bags_Under_Eyes', has_label=True, n_imgs=50)
z_unbag = get_features_for(label='Bags_Under_Eyes', has_label=False, n_imgs=50)

# Chubby features
z_chubby = get_features_for(label='Chubby', has_label=True, n_imgs=50)
z_unchubby = get_features_for(label='Chubby', has_label=False, n_imgs=50)

# Grey features
z_grey = get_features_for(label='Gray_Hair', has_label=True, n_imgs=50)
z_ungrey = get_features_for(label='Gray_Hair', has_label=False, n_imgs=50)

# Unblur vector
unblur_vector = z_unblur - z_blur
# Smile vector
smile_vector = z_smile - z_unsmile
# Attractive vector
attractive_vector = z_attractive - z_unattractive
# Young vector
young_vector = z_young - z_unyoung
# Eye Bags vector
bag_vector = z_bag - z_unbag
# Chubby vector
chubby_vector = z_chubby - z_unchubby
# Grey Bags vector
grey_vector = z_grey - z_ungrey

# Generate the images for the features
b_smile = sess.run(G, feed_dict={Z: z_smile[np.newaxis]})
b_unsmile = sess.run(G, feed_dict={Z: z_unsmile[np.newaxis]})
b_attractive = sess.run(G, feed_dict={Z: z_attractive[np.newaxis]})
b_unattractive = sess.run(G, feed_dict={Z: z_unattractive[np.newaxis]})
b_young = sess.run(G, feed_dict={Z: z_young[np.newaxis]})
b_unyoung = sess.run(G, feed_dict={Z: z_unyoung[np.newaxis]})
b_chubby = sess.run(G, feed_dict={Z: z_chubby[np.newaxis]})
b_unchubby = sess.run(G, feed_dict={Z: z_unchubby[np.newaxis]})
b_grey = sess.run(G, feed_dict={Z: z_grey[np.newaxis]})
b_ungrey = sess.run(G, feed_dict={Z: z_ungrey[np.newaxis]})
b_bag =  sess.run(G, feed_dict={Z: z_bag[np.newaxis]})
```

    C:\Users\eutan\Anaconda3\lib\site-packages\skimage\transform\_warps.py:84: UserWarning: The default mode, 'constant', will be changed to 'reflect' in skimage 0.15.
      warn("The default mode, 'constant', will be changed to 'reflect' in "
    

Plot the reconstructed latent information of Smile, Attractive, Young, Chubby and Grey


```python
fig, axs = plt.subplots(1, 11, figsize=(15, 6))
axs[0].imshow(b_smile[0] / b_smile.max()), axs[0].set_title('Smile'), axs[0].grid('off'), axs[0].axis('off')
axs[1].imshow(b_unsmile[0] / b_unsmile.max()), axs[1].set_title('Not Smile'), axs[1].grid('off'), axs[1].axis('off')
axs[2].imshow(b_attractive[0] / b_attractive.max()), axs[2].set_title('Attractive'), axs[2].grid('off'), axs[2].axis('off')
axs[3].imshow(b_unattractive[0] / b_unattractive.max()), axs[3].set_title('Not Attractive'), axs[3].grid('off'), axs[3].axis('off')
axs[4].imshow(b_young[0] / b_young.max()), axs[4].set_title('Young'), axs[4].grid('off'), axs[4].axis('off')
axs[5].imshow(b_unyoung[0] / b_unyoung.max()), axs[5].set_title('Not Young'), axs[5].grid('off'), axs[5].axis('off')
axs[6].imshow(b_chubby[0] / b_chubby.max()), axs[6].set_title('Chubby'), axs[6].grid('off'), axs[6].axis('off')
axs[7].imshow(b_unchubby[0] / b_unchubby.max()), axs[7].set_title('Not Chubby'), axs[7].grid('off'), axs[7].axis('off')
axs[8].imshow(b_grey[0] / b_grey.max()), axs[8].set_title('Grey'), axs[8].grid('off'), axs[8].axis('off')
axs[9].imshow(b_ungrey[0] / b_ungrey.max()), axs[9].set_title('Not Grey'), axs[9].grid('off'), axs[9].axis('off')
axs[10].imshow(b_bag[0] / b_bag.max()), axs[10].set_title('Eye Bag'), axs[10].grid('off'), axs[10].axis('off')
```




    (<matplotlib.image.AxesImage at 0x15c034aab70>,
     <matplotlib.text.Text at 0x15c03476ba8>,
     None,
     (-0.5, 99.5, 99.5, -0.5))




![png](readme_imgs/output_183_1.png)


Now synthesize the images from young and attractive, to grey and chubby


```python
n_imgs = 20
amt = np.linspace(0, 1.5, n_imgs)
z = sess.run(Z, feed_dict={X: x})

imgs = []
for amt_i in amt:
    zs = z + synthetic_unblur_vector * amt_i + amt_i * smile_vector + amt_i * attractive_vector
    g = sess.run(G, feed_dict={Z: zs})
    m = utils.montage(np.clip(g, 0, 1))
    imgs.append(m)

for amt_i in amt:
    zs2 = zs - (amt_i * smile_vector * 2) + amt_i * chubby_vector + (0.5 * amt_i * bag_vector) + (amt_i * grey_vector * 0.5) - (amt_i * young_vector * 0.5)
    g = sess.run(G, feed_dict={Z: zs2})
    m = utils.montage(np.clip(g, 0, 1))
    imgs.append(m)
```

Create the gif


```python
import imageio
imageio.mimsave("vaegan.gif", imgs)
ipyd.Image(url='vaegan.gif?i={}'.format(
            np.random.rand()), height=1000, width=1000)
```




<img src="vaegan.gif?i=0.9769104786044496" width="1000" height="1000"/>



<a name="part-4---character-level-recurrent-neural-network"></a>
# Part 4 - Character Level Recurrent Neural Network

Please visit [session-5-part2.ipynb](session-5-part2.ipynb) for the rest of the homework!