
# Autoencoders

This notebook was part of an assignment for an online MOOC - Creative Applications of Deep Learning with TensorFlow - Kadenze. The course material can be found at https://github.com/pkmital/CADL/tree/master/session-3, and the dataset can be found at https://github.com/yigher/deep_learning/blob/master/datasets/football_players.tar.gz.

It covers examples of a fully connected and CNN Autoencoder.

<table><img src='reconstruction1.gif'></td></tr></table>


Let's load the required libraries.


```python
# First check the Python version
import sys
if sys.version_info < (3,4):
    print('You are running an older version of Python!\n\n' \
          'You should consider updating to Python 3.4.0 or ' \
          'higher as the libraries built for this course ' \
          'have only been tested in Python 3.4 and higher.\n')
    print('Try installing the Python 3.5 version of anaconda '
          'and then restart `jupyter notebook`:\n' \
          'https://www.continuum.io/downloads\n\n')

# Now get necessary libraries
try:
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.transform import resize
    from skimage import data
    from scipy.misc import imresize
    import IPython.display as ipyd
except ImportError:
    print('You are missing some packages! ' \
          'We will try installing them before continuing!')
    !pip install "numpy>=1.11.0" "matplotlib>=1.5.1" "scikit-image>=0.11.3" "scikit-learn>=0.17" "scipy>=0.17.0"
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.transform import resize
    from skimage import data
    from scipy.misc import imresize
    import IPython.display as ipyd
    print('Done!')

# Import Tensorflow
try:
    import tensorflow as tf
except ImportError:
    print("You do not have tensorflow installed!")
    print("Follow the instructions on the following link")
    print("to install tensorflow before continuing:")
    print("")
    print("https://github.com/pkmital/CADL#installation-preliminaries")

# This cell includes the provided libraries from the zip file
# and a library for displaying images from ipython, which
# we will use to display the gif
try:
    from libs import utils, gif, datasets, dataset_utils, vae, dft
except ImportError:
    print("Make sure you have started notebook in the same directory" +
          " as the provided zip file which includes the 'libs' folder" +
          " and the file 'utils.py' inside of it.  You will NOT be able"
          " to complete this assignment unless you restart jupyter"
          " notebook inside the directory created by extracting"
          " the zip file or cloning the github repo.")

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



Load the images. Feel free to change the img_dir variable at the location where you have stored the images.


```python
# TODO - change the variable if required.
img_dir = "epl"

# function to get images from a directory
def get_imgs(dst, max_images=1000):
    """Load the first `max_images` images of the celeb dataset.

    Returns
    -------
    imgs : list of np.ndarray
        List of the first 100 images from the celeb dataset
    """
    files = [os.path.join(dst, fname) for fname in os.listdir(dst)]
    
    # Read every filename as an RGB image
    return [plt.imread(fname)[..., :3] for fname in files[:max_images]]

# See how this works w/ Celeb Images or try your own dataset instead:

# Read every filename as an RGB image
imgs = get_imgs(img_dir, max_images=1000)

# Crop every image to a square
imgs = [utils.imcrop_tosquare(img_i) for img_i in imgs]

# Then resize the square image to 100 x 100 pixels
imgs = [resize(img_i, (80, 80)) for img_i in imgs]


# Then convert the list of images to a 4d array (e.g. use np.array to convert a list to a 4d array):
Xs = np.array(imgs).astype(np.float32)

print(Xs.shape)

# Then convert the list of images to a 4d array (e.g. use np.array to convert a list to a 4d array):
# Xs = np.reshape(imgs, [-1, imgs.shape[1], imgs.shape[2], imgs.shape[3]])
# print("Xs.shape: ", Xs.shape)
# print(Xs.shape)
# assert(Xs.ndim == 4 and Xs.shape[1] <= 100 and Xs.shape[2] <= 100)
```

    C:\Users\eutan\Anaconda3\lib\site-packages\skimage\transform\_warps.py:84: UserWarning: The default mode, 'constant', will be changed to 'reflect' in skimage 0.15.
      warn("The default mode, 'constant', will be changed to 'reflect' in "
    

    (640, 80, 80, 3)
    

We'll now make use of a library that helps store this data.  It provides some interfaces for generating "batches" of data, as well as splitting the data into training, validation, and testing sets.  To use it, we pass in the data and optionally its labels.  If we don't have labels, we just pass in the data.  In the second half of this notebook, we'll explore using a dataset's labels as well.


```python
ds = datasets.Dataset(Xs)
# Uncomment the below code if you want to use CIFAR-10 dataset instead
#ds = datasets.CIFAR10(flatten=False)
```

It allows us to easily find the mean:


```python
mean_img = ds.mean()
plt.imshow(mean_img)
# If your image comes out entirely black, try w/o the `astype(np.uint8)`
# that means your images are read in as 0-255, rather than 0-1 and 
# this simply depends on the version of matplotlib you are using.
```




    <matplotlib.image.AxesImage at 0x273d180cdd8>




![png](md_imgs/output_8_1.png)


Or the deviation:


```python
std_img = ds.std()
plt.imshow(std_img)
print(std_img.shape)
```

    (80, 80, 3)
    


![png](md_imgs/output_10_1.png)


Recall we can calculate the mean of the standard deviation across each color channel:


```python
std_img = np.mean(std_img, axis=2)
plt.imshow(std_img)
```




    <matplotlib.image.AxesImage at 0x273d19bd898>




![png](md_imgs/output_12_1.png)


All the input data we gave as input to our `Datasets` object, previously stored in `Xs` is now stored in a variable as part of our `ds` Datasets object, `X`:


```python
plt.imshow(ds.X[0])
print(ds.X.shape)
```

    (640, 80, 80, 3)
    


![png](md_imgs/output_14_1.png)


It takes a parameter, `split` at the time of creation, which allows us to create train/valid/test sets.  By default, this is set to `[1.0, 0.0, 0.0]`, which means to take all the data in the train set, and nothing in the validation and testing sets.  We can access "batch generators" of each of these sets by saying: `ds.train.next_batch`.  A generator is a really powerful way of handling iteration in Python.  If you are unfamiliar with the idea of generators, I recommend reading up a little bit on it, e.g. here: http://intermediatepythonista.com/python-generators - think of it as a for loop, but as a function.  It returns one iteration of the loop each time you call it.

This generator will automatically handle the randomization of the dataset.  Let's try looping over the dataset using the batch generator:


```python
for (X, y) in ds.train.next_batch(batch_size=100):
    print(X.shape)
```

    (100, 80, 80, 3)
    (100, 80, 80, 3)
    (100, 80, 80, 3)
    (100, 80, 80, 3)
    (100, 80, 80, 3)
    (100, 80, 80, 3)
    (40, 80, 80, 3)
    

This returns `X` and `y` as a tuple.  Since we're not using labels, we'll just ignore this.  The `next_batch` method takes a parameter, `batch_size`, which we'll set appropriately to our batch size.  Notice it runs for exactly 10 iterations to iterate over our 100 examples, then the loop exits.  The order in which it iterates over the 100 examples is randomized each time you iterate.

Write two functions to preprocess (normalize) any given image, and to unprocess it, i.e. unnormalize it by removing the normalization.  The `preprocess` function should perform exactly the task you learned to do in assignment 1: subtract the mean, then divide by the standard deviation.  The `deprocess` function should take the preprocessed image and undo the preprocessing steps.  Recall that the `ds` object contains the `mean` and `std` functions for access the mean and standarad deviation.  We'll be using the `preprocess` and `deprocess` functions on the input and outputs of the network.  Note, we could use Tensorflow to do this instead of numpy, but for sake of clarity, I'm keeping this separate from the Tensorflow graph.

We're going to now work on creating an autoencoder.  To start, we'll only use linear connections, like in the last assignment.  This means, we need a 2-dimensional input:  Batch Size x Number of Features.  We currently have a 4-dimensional input: Batch Size x Height x Width x Channels.  We'll have to calculate the number of features we have to help construct the Tensorflow Graph for our autoencoder neural network.  Then, when we are ready to train the network, we'll reshape our 4-dimensional dataset into a 2-dimensional one when feeding the input of the network.  Optionally, we could create a `tf.reshape` as the first operation of the network, so that we can still pass in our 4-dimensional array, and the Tensorflow graph would reshape it for us.  We'll try the former method, by reshaping manually, and then you can explore the latter method, of handling 4-dimensional inputs on your own.




```python
# Write a function to preprocess/normalize an image, given its dataset object
# (which stores the mean and standard deviation!)
def preprocess(img, ds):
    norm_img = (img - ds.mean()) / (ds.std() + 1e-10)
    return norm_img

# Write a function to undo the normalization of an image, given its dataset object
# (which stores the mean and standard deviation!)
def deprocess(norm_img, ds):
    img = norm_img * (ds.std() + 1e-10) + ds.mean()
    return img

for (X, y) in ds.train.next_batch(batch_size=100):
    norm_imgs = preprocess(X, ds)
    denorm_imgs = deprocess(norm_imgs, ds)
    norm_imgs = utils.montage(norm_imgs)
    denorm_imgs = utils.montage(denorm_imgs)
    fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    axs[0].imshow(norm_imgs)
    axs[0].set_title('Normalised')
    axs[1].imshow(denorm_imgs)
    axs[1].set_title('De-normalised')
    fig.canvas.draw()
    plt.show()
    break
```


![png](md_imgs/output_19_0.png)



```python
# Calculate the number of features in your image.
# This is the total number of pixels, or (height x width x channels).
n_features = ds.X.shape[1] * ds.X.shape[2] * ds.X.shape[3]
print(n_features)
```

    19200
    

Let's create a list of how many neurons we want in each layer.  This should be for just one half of the network, the encoder only.  It should start large, then get smaller and smaller.  We're also going to try an encode our dataset to an inner layer of just 2 values.  So from our number of features, we'll go all the way down to expressing that image by just 2 values.  Try a small network to begin with, then explore deeper networks:




```python
encoder_dimensions = [1024, 128, 30, 2]
```

Now create a placeholder just like in the last session in the tensorflow graph that will be able to get any number (None) of `n_features` inputs.



```python
from tensorflow.python.framework.ops import reset_default_graph
reset_default_graph()

X = tf.placeholder(tf.float32, shape=[None, n_features], name='X')
                   
assert(X.get_shape().as_list() == [None, n_features])
```

Now complete the function `encode` below.  This takes as input our input placeholder, `X`, our list of `dimensions`, and an `activation` function, e.g. `tf.nn.relu` or `tf.nn.tanh`, to apply to each layer's output, and creates a series of fully connected layers.  This works just like in the last session!  We multiply our input, add a bias, then apply a non-linearity.  Instead of having 20 neurons in each layer, we're going to use our `dimensions` list to tell us how many neurons we want in each layer.

One important difference is that we're going to also store every weight matrix we create!  This is so that we can use the same weight matrices when we go to build our decoder.  This is a *very* powerful concept that creeps up in a few different neural network architectures called weight sharing.  Weight sharing isn't necessary to do of course, but can speed up training and offer a different set of features depending on your dataset.  Explore trying both.  We'll also see how another form of weight sharing works in convolutional networks.




```python
def encode(X, dimensions, activation=tf.nn.relu):
    # We're going to keep every matrix we create so let's create a list to hold them all
    Ws = []

    # We'll create a for loop to create each layer:
    for layer_i, n_output in enumerate(dimensions):

        # TODO: just like in the last session,
        # we'll use a variable scope to help encapsulate our variables
        # This will simply prefix all the variables made in this scope
        # with the name we give it.  Make sure it is a unique name
        # for each layer, e.g., 'encoder/layer1', 'encoder/layer2', or
        # 'encoder/1', 'encoder/2',... 
        
        with tf.variable_scope('encoder/layer{}'.format(layer_i)):

            # TODO: Create a weight matrix which will increasingly reduce
            # down the amount of information in the input by performing
            # a matrix multiplication.  You can use the utils.linear function.
            h, W = utils.linear(X, n_output, activation = tf.nn.relu)
            
            # TODO: Apply an activation function (unless you used the parameter
            # for activation function in the utils.linear call)

            # Finally we'll store the weight matrix.
            # We need to keep track of all
            # the weight matrices we've used in our encoder
            # so that we can build the decoder using the
            # same weight matrices.
            Ws.append(W)
            
            # Replace X with the current layer's output, so we can
            # use it in the next layer.
            X = h
    
    z = X
    return Ws, z
```

We now have a function for encoding an input `X`.  Take note of which activation function you use as this will be important for the behavior of the latent encoding, `z`, later on.


```python
# Then call the function
Ws, z = encode(X, encoder_dimensions)

# And just some checks to make sure you've done it right.
assert(z.get_shape().as_list() == [None, 2])
assert(len(Ws) == len(encoder_dimensions))
```

Let's take a look at the graph:


```python
[op.name for op in tf.get_default_graph().get_operations()]
```




    ['X',
     'encoder/layer0/fc/W/Initializer/random_uniform/shape',
     'encoder/layer0/fc/W/Initializer/random_uniform/min',
     'encoder/layer0/fc/W/Initializer/random_uniform/max',
     'encoder/layer0/fc/W/Initializer/random_uniform/RandomUniform',
     'encoder/layer0/fc/W/Initializer/random_uniform/sub',
     'encoder/layer0/fc/W/Initializer/random_uniform/mul',
     'encoder/layer0/fc/W/Initializer/random_uniform',
     'encoder/layer0/fc/W',
     'encoder/layer0/fc/W/Assign',
     'encoder/layer0/fc/W/read',
     'encoder/layer0/fc/b/Initializer/Const',
     'encoder/layer0/fc/b',
     'encoder/layer0/fc/b/Assign',
     'encoder/layer0/fc/b/read',
     'encoder/layer0/fc/MatMul',
     'encoder/layer0/fc/h',
     'encoder/layer0/fc/Relu',
     'encoder/layer1/fc/W/Initializer/random_uniform/shape',
     'encoder/layer1/fc/W/Initializer/random_uniform/min',
     'encoder/layer1/fc/W/Initializer/random_uniform/max',
     'encoder/layer1/fc/W/Initializer/random_uniform/RandomUniform',
     'encoder/layer1/fc/W/Initializer/random_uniform/sub',
     'encoder/layer1/fc/W/Initializer/random_uniform/mul',
     'encoder/layer1/fc/W/Initializer/random_uniform',
     'encoder/layer1/fc/W',
     'encoder/layer1/fc/W/Assign',
     'encoder/layer1/fc/W/read',
     'encoder/layer1/fc/b/Initializer/Const',
     'encoder/layer1/fc/b',
     'encoder/layer1/fc/b/Assign',
     'encoder/layer1/fc/b/read',
     'encoder/layer1/fc/MatMul',
     'encoder/layer1/fc/h',
     'encoder/layer1/fc/Relu',
     'encoder/layer2/fc/W/Initializer/random_uniform/shape',
     'encoder/layer2/fc/W/Initializer/random_uniform/min',
     'encoder/layer2/fc/W/Initializer/random_uniform/max',
     'encoder/layer2/fc/W/Initializer/random_uniform/RandomUniform',
     'encoder/layer2/fc/W/Initializer/random_uniform/sub',
     'encoder/layer2/fc/W/Initializer/random_uniform/mul',
     'encoder/layer2/fc/W/Initializer/random_uniform',
     'encoder/layer2/fc/W',
     'encoder/layer2/fc/W/Assign',
     'encoder/layer2/fc/W/read',
     'encoder/layer2/fc/b/Initializer/Const',
     'encoder/layer2/fc/b',
     'encoder/layer2/fc/b/Assign',
     'encoder/layer2/fc/b/read',
     'encoder/layer2/fc/MatMul',
     'encoder/layer2/fc/h',
     'encoder/layer2/fc/Relu',
     'encoder/layer3/fc/W/Initializer/random_uniform/shape',
     'encoder/layer3/fc/W/Initializer/random_uniform/min',
     'encoder/layer3/fc/W/Initializer/random_uniform/max',
     'encoder/layer3/fc/W/Initializer/random_uniform/RandomUniform',
     'encoder/layer3/fc/W/Initializer/random_uniform/sub',
     'encoder/layer3/fc/W/Initializer/random_uniform/mul',
     'encoder/layer3/fc/W/Initializer/random_uniform',
     'encoder/layer3/fc/W',
     'encoder/layer3/fc/W/Assign',
     'encoder/layer3/fc/W/read',
     'encoder/layer3/fc/b/Initializer/Const',
     'encoder/layer3/fc/b',
     'encoder/layer3/fc/b/Assign',
     'encoder/layer3/fc/b/read',
     'encoder/layer3/fc/MatMul',
     'encoder/layer3/fc/h',
     'encoder/layer3/fc/Relu']



So we've created a few layers, encoding our input `X` all the way down to 2 values in the tensor `z`.  We do this by multiplying our input `X` by a set of matrices shaped as:


```python
[W_i.get_shape().as_list() for W_i in Ws]
```




    [[19200, 1024], [1024, 128], [128, 30], [30, 2]]



Resulting in a layer which is shaped as:


```python
z.get_shape().as_list()
```




    [None, 2]



## Building the Decoder 

Here is a helpful animation on what the matrix "transpose" operation does:
![transpose](https://upload.wikimedia.org/wikipedia/commons/e/e4/Matrix_transpose.gif)

Basically what is happening is rows becomes columns, and vice-versa.  We're going to use our existing weight matrices but transpose them so that we can go in the opposite direction.  In order to build our decoder, we'll have to do the opposite of what we've just done, multiplying `z` by the transpose of our weight matrices, to get back to a reconstructed version of `X`.  First, we'll reverse the order of our weight matrics, and then append to the list of dimensions the final output layer's shape to match our input:


```python
# We'll first reverse the order of our weight matrices
decoder_Ws = Ws[::-1]

# then reverse the order of our dimensions
# appending the last layers number of inputs.
decoder_dimensions = encoder_dimensions[::-1][1:] + [n_features]
print(decoder_dimensions)

assert(decoder_dimensions[-1] == n_features)
```

    [30, 128, 1024, 19200]
    

Now we'll build the decoder.  I've shown you how to do this.  Read through the code to fully understand what it is doing:


```python
def decode(z, dimensions, Ws, activation=tf.nn.relu):
    current_input = z
    for layer_i, n_output in enumerate(dimensions):
        # we'll use a variable scope again to help encapsulate our variables
        # This will simply prefix all the variables made in this scope
        # with the name we give it.
        with tf.variable_scope("decoder/layer/{}".format(layer_i)):

            # Now we'll grab the weight matrix we created before and transpose it
            # So a 3072 x 784 matrix would become 784 x 3072
            # or a 256 x 64 matrix, would become 64 x 256
            W = tf.transpose(Ws[layer_i])

            # Now we'll multiply our input by our transposed W matrix
            h = tf.matmul(current_input, W)

            # And then use a relu activation function on its output
            current_input = activation(h)

            # We'll also replace n_input with the current n_output, so that on the
            # next iteration, our new number inputs will be correct.
            n_input = n_output
    Y = current_input
    return Y
```


```python
Y = decode(z, decoder_dimensions, decoder_Ws)
```

Let's take a look at the new operations we've just added.  They will all be prefixed by "decoder" so we can use list comprehension to help us with this:


```python
[op.name for op in tf.get_default_graph().get_operations()
 if op.name.startswith('decoder')]
```




    ['decoder/layer/0/transpose/Rank',
     'decoder/layer/0/transpose/sub/y',
     'decoder/layer/0/transpose/sub',
     'decoder/layer/0/transpose/Range/start',
     'decoder/layer/0/transpose/Range/delta',
     'decoder/layer/0/transpose/Range',
     'decoder/layer/0/transpose/sub_1',
     'decoder/layer/0/transpose',
     'decoder/layer/0/MatMul',
     'decoder/layer/0/Relu',
     'decoder/layer/1/transpose/Rank',
     'decoder/layer/1/transpose/sub/y',
     'decoder/layer/1/transpose/sub',
     'decoder/layer/1/transpose/Range/start',
     'decoder/layer/1/transpose/Range/delta',
     'decoder/layer/1/transpose/Range',
     'decoder/layer/1/transpose/sub_1',
     'decoder/layer/1/transpose',
     'decoder/layer/1/MatMul',
     'decoder/layer/1/Relu',
     'decoder/layer/2/transpose/Rank',
     'decoder/layer/2/transpose/sub/y',
     'decoder/layer/2/transpose/sub',
     'decoder/layer/2/transpose/Range/start',
     'decoder/layer/2/transpose/Range/delta',
     'decoder/layer/2/transpose/Range',
     'decoder/layer/2/transpose/sub_1',
     'decoder/layer/2/transpose',
     'decoder/layer/2/MatMul',
     'decoder/layer/2/Relu',
     'decoder/layer/3/transpose/Rank',
     'decoder/layer/3/transpose/sub/y',
     'decoder/layer/3/transpose/sub',
     'decoder/layer/3/transpose/Range/start',
     'decoder/layer/3/transpose/Range/delta',
     'decoder/layer/3/transpose/Range',
     'decoder/layer/3/transpose/sub_1',
     'decoder/layer/3/transpose',
     'decoder/layer/3/MatMul',
     'decoder/layer/3/Relu']



And let's take a look at the output of the autoencoder:


```python
Y.get_shape().as_list()
```




    [None, 19200]



Great!  So we should have a synthesized version of our input placeholder, `X`, inside of `Y`.  This `Y` is the result of many matrix multiplications, first a series of multiplications in our encoder all the way down to 2 dimensions, and then back to the original dimensions through our decoder.  Let's now create a pixel-to-pixel measure of error.  This should measure the difference in our synthesized output, `Y`, and our input, `X`.  You can use the $l_1$ or $l_2$ norm, just like in assignment 2.  If you don't remember, go back to homework 2 where we calculated the cost function and try the same idea here.



```python
# Calculate some measure of loss, e.g. the pixel to pixel absolute difference or squared difference
loss = tf.squared_difference(X, Y)

# Now sum over every pixel and then calculate the mean over the batch dimension (just like session 2!)
# hint, use tf.reduce_mean and tf.reduce_sum
cost = tf.reduce_mean(tf.reduce_sum(loss, 1))
```

Now for the standard training code.  We'll pass our `cost` to an optimizer, and then use mini batch gradient descent to optimize our network's parameters.  We just have to be careful to make sure we're preprocessing our input and feed it in the right shape, a 2-dimensional matrix of [batch_size, n_features] in dimensions.




```python
learning_rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
```

Below is the training code for our autoencoder.  Please go through each line of code to make sure you understand what is happening, and fill in the missing pieces.  This will take awhile.  On my machine, it takes about 15 minutes.  If you're impatient, you can "Interrupt" the kernel by going to the Kernel menu above, and continue with the notebook.  Though, the longer you leave this to train, the better the result will be.

What I really want you to notice is what the network learns to encode first, based on what it is able to reconstruct.  It won't able to reconstruct everything.  At first, it will just be the mean image.  Then, other major changes in the dataset.  For the first 100 images of celeb net, this seems to be the background: white, blue, black backgrounds.  From this basic interpretation, you can reason that the autoencoder has learned a representation of the backgrounds, and is able to encode that knowledge of the background in its inner most layer of just two values.  It then goes on to represent the major variations in skin tone and hair.  Then perhaps some facial features such as lips.  So the features it is able to encode tend to be the major things at first, then the smaller things.




```python
# (TODO) Create a tensorflow session and initialize all of our weights:
sess = tf.Session()
sess.run(tf.global_variables_initializer())
```

Note that if you run into "InternalError" or "ResourceExhaustedError", it is likely that you have run out of memory!  Try a smaller network!  For instance, restart the notebook's kernel, and then go back to defining `encoder_dimensions = [256, 2]` instead.  If you run into memory problems below, you can also try changing the batch_size to 50.


```python
# Some parameters for training
batch_size = 100
n_epochs = 300
step = 10

# We'll try to reconstruct the same first 100 images and show how
# The network does over the course of training.
examples = ds.X[:100]

# We have to preprocess the images before feeding them to the network.
# I'll do this once here, so we don't have to do it every iteration.
test_examples = preprocess(examples, ds).reshape(-1, n_features)

# If we want to just visualize them, we can create a montage.
test_images = utils.montage(examples)

# Store images so we can make a gif
gifs = []

# Now for our training:
for epoch_i in range(n_epochs):
    
    # Keep track of the cost
    this_cost = 0
    
    # Iterate over the entire dataset in batches
    for batch_X, _ in ds.train.next_batch(batch_size=batch_size):
        
        # (TODO) Preprocess and reshape our current batch, batch_X:
        this_batch = preprocess(batch_X, ds).reshape(-1, n_features)
        
        # Compute the cost, and run the optimizer.
        this_cost += sess.run([cost, optimizer], feed_dict={X: this_batch})[0]
    
    # Average cost of this epoch
    avg_cost = this_cost / ds.X.shape[0] / batch_size
    print(epoch_i, avg_cost)
    
    # Let's also try to see how the network currently reconstructs the input.
    # We'll draw the reconstruction every `step` iterations.
    if epoch_i % step == 0:
        
        # (TODO) Ask for the output of the network, Y, and give it our test examples
        recon = sess.run(Y, feed_dict={X: test_examples})
                         
        # Resize the 2d to the 4d representation:
        rsz = recon.reshape(examples.shape)

        # We have to unprocess the image now, removing the normalization
        unnorm_img = deprocess(rsz, ds)
                         
        # Clip to avoid saturation
        # TODO: Make sure this image is the correct range, e.g.
        # for float32 0-1, you should clip between 0 and 1
        # for uint8 0-255, you should clip between 0 and 255!
        clipped = np.clip(unnorm_img, 0, 255)

        # And we can create a montage of the reconstruction
        recon = utils.montage(clipped)
        
        # Store for gif
        gifs.append(recon)

        fig, axs = plt.subplots(1, 2, figsize=(10, 10))
        axs[0].imshow(test_images)
        axs[0].set_title('Original')
        axs[1].imshow(recon)
        axs[1].set_title('Synthesis')
        fig.canvas.draw()
        plt.show()
```

    0 2.11383166504
    


![png](md_imgs/output_52_1.png)


    1 2.00287155151
    2 1.98101719666
    3 2.00982748413
    4 1.91053456116
    5 1.92314825439
    6 1.93306268311
    7 1.95458401489
    8 1.91254042053
    9 1.9038291626
    10 1.92234623718
    
![png](md_imgs/output_52_59.png)


    291 1.76637063599
    292 1.72359309387
    293 1.73033711243
    294 1.73919445801
    295 1.72832299805
    296 1.78352661133
    297 1.76846618652
    298 1.74568348694
    299 1.75053509521
    

Let's take a look a the final reconstruction:


```python
fig, axs = plt.subplots(1, 2, figsize=(10, 10))
axs[0].imshow(test_images)
axs[0].set_title('Original')
axs[1].imshow(recon)
axs[1].set_title('Synthesis')
fig.canvas.draw()
plt.show()
plt.imsave(arr=test_images, fname='test.png')
plt.imsave(arr=recon, fname='recon.png')
```


![png](md_imgs/output_54_0.png)


<a name="visualize-the-embedding"></a>
## Visualize the Embedding

Let's now try visualizing our dataset's inner most layer's activations.  Since these are already 2-dimensional, we can use the values of this layer to position any input image in a 2-dimensional space.  We hope to find similar looking images closer together.

We'll first ask for the inner most layer's activations when given our example images.  This will run our images through the network, half way, stopping at the end of the encoder part of the network.


```python
zs = sess.run(z, feed_dict={X:test_examples})
```

Recall that this layer has 2 neurons:


```python
zs.shape
```




    (100, 2)



Let's see what the activations look like for our 100 images as a scatter plot.


```python
plt.scatter(zs[:, 0], zs[:, 1])
```




    <matplotlib.collections.PathCollection at 0x7f45980d8978>




![png](md_imgs/output_60_1.png)


If you view this plot over time, and let the process train longer, you will see something similar to the visualization here on the right: https://vimeo.com/155061675 - the manifold is able to express more and more possible ideas, or put another way, it is able to encode more data. As it grows more expressive, with more data, and longer training, or deeper networks, it will fill in more of the space, and have different modes expressing different clusters of the data.  With just 100 examples of our dataset, this is *very* small to try to model with such a deep network.  In any case, the techniques we've learned up to now apply in exactly the same way, even if we had 1k, 100k, or even many millions of images.

Let's try to see how this minimal example, with just 100 images, and just 100 epochs looks when we use this embedding to sort our dataset, just like we tried to do in the 1st assignment, but now with our autoencoders embedding.

<a name="reorganize-to-grid"></a>
## Reorganize to Grid

We'll use these points to try to find an assignment to a grid.  This is a well-known problem known as the "assignment problem": https://en.wikipedia.org/wiki/Assignment_problem - This is unrelated to the applications we're investigating in this course, but I thought it would be a fun extra to show you how to do.  What we're going to do is take our scatter plot above, and find the best way to stretch and scale it so that each point is placed in a grid.  We try to do this in a way that keeps nearby points close together when they are reassigned in their grid.


```python
n_images = 100
idxs = np.linspace(np.min(zs) * 2.0, np.max(zs) * 2.0,
                   int(np.ceil(np.sqrt(n_images))))
xs, ys = np.meshgrid(idxs, idxs)
grid = np.dstack((ys, xs)).reshape(-1, 2)[:n_images,:]
```


```python
fig, axs = plt.subplots(1,2,figsize=(8,3))
axs[0].scatter(zs[:, 0], zs[:, 1],
               edgecolors='none', marker='o', s=2)
axs[0].set_title('Autoencoder Embedding')
axs[1].scatter(grid[:,0], grid[:,1],
               edgecolors='none', marker='o', s=2)
axs[1].set_title('Ideal Grid')
```




    <matplotlib.text.Text at 0x7f459873e710>




![png](md_imgs/output_63_1.png)


To do this, we can use scipy and an algorithm for solving this assignment problem known as the hungarian algorithm.  With a few points, this algorithm runs pretty fast.  But be careful if you have many more points, e.g. > 1000, as it is not a very efficient algorithm!


```python
from scipy.spatial.distance import cdist
cost = cdist(grid[:, :], zs[:, :], 'sqeuclidean')
from scipy.optimize._hungarian import linear_sum_assignment
indexes = linear_sum_assignment(cost)
```

The result tells us the matching indexes from our autoencoder embedding of 2 dimensions, to our idealized grid:


```python
indexes
```




    (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
            34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
            51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
            68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
            85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]),
     array([96, 92, 85, 73, 53, 79, 68, 20, 39, 75, 67, 66, 65, 61, 23, 80, 16,
            51, 97, 36, 60, 59, 57, 55, 72, 64, 46, 52,  9,  1, 50, 47, 42, 37,
            77, 89, 78, 94,  0,  5, 29, 26, 18, 11, 25, 58, 54, 35, 91,  8, 10,
             6,  4,  3, 31, 45,  2, 22, 30, 38, 70, 33, 84, 71, 12, 21, 69, 19,
            17, 43, 76, 32, 63, 74, 24, 27, 99, 82, 83, 49, 28, 41, 88, 44, 93,
            98, 48, 81, 86, 87, 90, 56,  7, 95, 40, 13, 14, 34, 62, 15]))




```python
plt.figure(figsize=(5, 5))
for i in range(len(zs)):
    plt.plot([zs[indexes[1][i], 0], grid[i, 0]],
             [zs[indexes[1][i], 1], grid[i, 1]], 'r')
plt.xlim([-3, 3])
plt.ylim([-3, 3])
```




    (-3, 3)




![png](md_imgs/output_68_1.png)


In other words, this algorithm has just found the best arrangement of our previous `zs` as a grid.  We can now plot our images using the order of our assignment problem to see what it looks like:


```python
examples_sorted = []
for i in indexes[1]:
    examples_sorted.append(examples[i])
plt.figure(figsize=(15, 15))
img = utils.montage(np.array(examples_sorted))
plt.imshow(img,
           interpolation='nearest')
plt.imsave(arr=img, fname='sorted.png')
```


![png](md_imgs/output_70_0.png)


<a name="2d-latent-manifold"></a>
## 2D Latent Manifold


We'll now explore the inner most layer of the network.  Recall we go from the number of image features (the number of pixels), down to 2 values using successive matrix multiplications, back to the number of image features through more matrix multiplications.  These inner 2 values are enough to represent our entire dataset (+ some loss, depending on how well we did).  Let's explore how the decoder, the second half of the network, operates, from just these two values.  We'll bypass the input placeholder, X, and the entire encoder network, and start from Z.  Let's first get some data which will sample Z in 2 dimensions from -1 to 1.  This range may be different for you depending on what your latent space's range of values are.  You can try looking at the activations for your `z` variable for a set of test images, as we've done before, and look at the range of these values.  Or try to guess based on what activation function you may have used on the `z` variable, if any.  

Then we'll use this range to create a linear interpolation of latent values, and feed these values through the decoder network to have our synthesized images to see what they look like.


```python
# This is a quick way to do what we could have done as
# a nested for loop:
zs = np.meshgrid(np.linspace(-1, 1, 10),
                 np.linspace(-1, 1, 10))

# Now we have 100 x 2 values of every possible position
# in a 2D grid from -1 to 1:
zs = np.c_[zs[0].ravel(), zs[1].ravel()]
```

Now calculate the reconstructed images using our new zs.  You'll want to start from the beginning of the decoder!  That is the `z` variable!  Then calculate the `Y` given our synthetic values for `z` stored in `zs`. 




```python
recon = sess.run(Y, feed_dict={z: zs})

# reshape the result to an image:
rsz = recon.reshape(examples.shape)

# Deprocess the result, unnormalizing it
unnorm_img = deprocess(rsz, ds)

# clip to avoid saturation
clipped = np.clip(unnorm_img, 0, 255)

# Create a montage
img_i = utils.montage(clipped)
```

And now we can plot the reconstructed montage representing our latent space:


```python
plt.figure(figsize=(15, 15))
plt.imshow(img_i)
plt.imsave(arr=img_i, fname='manifold.png')
```


![png](md_imgs/output_76_0.png)


<a name="part-two---general-autoencoder-framework"></a>
# Part Two - General Autoencoder Framework

There are a number of extensions we can explore w/ an autoencoder.  I've provided a module under the libs folder, `vae.py`, which you will need to explore for Part Two.  It has a function, `VAE`, to create an autoencoder, optionally with Convolution, Denoising, and/or Variational Layers.  Please read through the documentation and try to understand the different parameters.    

I've also included three examples of how to use the `VAE(...)` and `train_vae(...)` functions.  First look at the one using MNIST.  Then look at the other two: one using the Celeb Dataset; and lastly one which will download Sita Sings the Blues, rip the frames, and train a Variational Autoencoder on it.  This last one requires `ffmpeg` be installed (e.g. for OSX users, `brew install ffmpeg`, Linux users, `sudo apt-get ffmpeg-dev`, or else: https://ffmpeg.org/download.html).  The Celeb and Sita Sings the Blues training require us to use an image pipeline, which I've mentioned briefly during the lecture.  This does many things for us: it loads data from disk in batches, decodes the data as an image, resizes/crops the image, and uses a multithreaded graph to handle it all.  It is *very* efficient and is the way to go when handling large image datasets.  

The MNIST training does not use this.  Instead, the entire dataset is loaded into the CPU memory, and then fed in minibatches to the graph using Python/Numpy.  This is far less efficient, but will not be an issue for such a small dataset, e.g.  70k examples of 28x28 pixels = ~1.6 MB of data, easily fits into memory (in fact, it would really be better to use a Tensorflow variable with this entire dataset defined).  When you consider the Celeb Net, you have 200k examples of 218x178x3 pixels = ~700 MB of data.  That's just for the dataset.  When you factor in everything required for the network and its weights, then you are pushing it.  Basically this image pipeline will handle loading the data from disk, rather than storing it in memory.

<a name="instructions-1"></a>
## Instructions

You'll now try to train your own autoencoder using this framework.  You'll need to get a directory full of 'jpg' files.  You'll then use the VAE framework and the `vae.train_vae` function to train a variational autoencoder on your own dataset.  This accepts a list of files, and will output images of the training in the same directory.  These are named "test_xs.png" as well as many images named prefixed by "manifold" and "reconstruction" for each iteration of the training.  After you are happy with your training, you will need to create a forum post with the "test_xs.png" and the very last manifold and reconstruction image created to demonstrate how the variational autoencoder worked for your dataset.  You'll likely need a lot more than 100 images for this to be successful.

Note that this will also create "checkpoints" which save the model!  If you change the model, and already have a checkpoint by the same name, it will try to load the previous model and will fail.  Be sure to remove the old checkpoint or specify a new name for `ckpt_name`!  The default parameters shown below are what I have used for the celeb net dataset which has over 200k images.  You will definitely want to use a smaller model if you do not have this many images!  Explore!




```python
# Get a list of jpg file (Only JPG works!)
some_dir='epl'
files = [os.path.join(some_dir, file_i) for file_i in os.listdir(some_dir) if file_i.endswith('.jpg')]

from tensorflow.python.framework.ops import reset_default_graph
reset_default_graph()

# Ensure that you have the latest TensorFlow version installed, otherwise you may have encountered
# 'rsz_shape' error because of the backward incompatible API.
# Train it!  Change these parameters!
vae.train_vae(files,
              input_shape=[80, 80, 3],
              learning_rate=0.0001,
              batch_size=100,
              n_epochs=20,
              n_examples=10,
              crop_shape=[64, 64, 3],
              crop_factor=0.8,
              n_filters=[32, 64, 128],
              n_hidden=256,
              n_code=50,
              convolutional=True,
              variational=True,
              filter_sizes=[3, 3, 3],
              dropout=True,
              keep_prob=0.8,
              activation=tf.nn.relu,
              img_step=100,
              save_step=100,
              ckpt_name="vae.ckpt")
```

Below are the test images and a gif of the trained model's prediction on the test images over time.


```python
HTML("<table><img src='reconstruction1.gif'></td></tr></table>")
```




<table><img src='reconstruction1.gif'></td></tr></table>


