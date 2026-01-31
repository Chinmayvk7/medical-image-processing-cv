'''
Autoencoder that removes noise in an image
'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# Implementation from scratch.

class MyConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, activation=None, padding='same', **kwargs):
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.kernel_size = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)
        self.activation = tf.keras.activations.get(activation)
        self.padding = padding.upper()

    def build(self, input_shape):
        # input_shape: (batch, H, W, C)
        in_channels = int(input_shape[-1])
        kh, kw = self.kernel_size
        # Kernel shape for tf.nn.conv2d: [kh, kw, in_channels, out_channels]
        self.kernel = self.add_weight(
            name='kernel',
            shape=(kh, kw, in_channels, self.filters),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
            dtype=self.dtype
        )
        self.bias = self.add_weight(
            name='bias',
            shape=(self.filters,),
            initializer='zeros',
            trainable=True,
            dtype=self.dtype
        )

    def call(self, inputs):
        # apply convolution (strides = 1)
        x = tf.nn.conv2d(inputs, self.kernel, strides=[1, 1, 1, 1], padding=self.padding)
        x = tf.nn.bias_add(x, self.bias)
        if self.activation is not None:
            x = self.activation(x)
        return x


class MyMaxPool2D(tf.keras.layers.Layer):
    def __init__(self, pool_size=(2, 2), padding='same', **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size if isinstance(pool_size, (tuple, list)) else (pool_size, pool_size)
        self.padding = padding.upper()

    def call(self, inputs):
        ksize = [self.pool_size[0], self.pool_size[1]]
        strides = [self.pool_size[0], self.pool_size[1]]
        # tf.nn.max_pool2d accepts ksize and strides as lists of two ints
        return tf.nn.max_pool2d(inputs, ksize=ksize, strides=strides, padding=self.padding)


class MyUpSampling2D(tf.keras.layers.Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super().__init__(**kwargs)
        self.size = size if isinstance(size, (tuple, list)) else (size, size)

    def call(self, inputs):
        # Nearest-neighbor upsampling implemented with tf.repeat
        x = tf.repeat(inputs, repeats=self.size[0], axis=1)
        x = tf.repeat(x, repeats=self.size[1], axis=2)
        return x
# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Filter the dataset to include only '2's
x_train_2s = x_train[y_train == 2]
x_test_2s = x_test[y_test == 2]

# Normalize the data
x_train_2s = x_train_2s.astype('float32') / 255.
x_test_2s = x_test_2s.astype('float32') / 255.

# Reshape the data to include the channel dimension
x_train_2s = np.reshape(x_train_2s, (len(x_train_2s), 28, 28, 1))
x_test_2s = np.reshape(x_test_2s, (len(x_test_2s), 28, 28, 1))

# Add noise to the images
noise_factor = 0.3
x_train_noisy = x_train_2s + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train_2s.shape) 
x_test_noisy = x_test_2s + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test_2s.shape)

# Make sure all values are between 0 and 1
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Build the autoencoder
input_img = layers.Input(shape=(28, 28, 1))

# Encoder
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = models.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder
autoencoder.fit(x_train_noisy, x_train_2s,epochs=50,batch_size=128)

# Use the autoencoder to denoise the test images
decoded_imgs = autoencoder.predict(x_test_noisy)

n = 4
plt.figure(figsize=(8, 10))

for i in range(n):
    # Original
    ax = plt.subplot(n, 3, 3*i + 1)
    ax.imshow(x_test_2s[i].reshape(28, 28), cmap='gray')
    ax.set_title("Original")
    ax.axis('off')

    # Noisy
    ax = plt.subplot(n, 3, 3*i + 2)
    ax.imshow(x_test_noisy[i].reshape(28, 28), cmap='gray')
    ax.set_title("Noisy")
    ax.axis('off')

    # Denoised
    ax = plt.subplot(n, 3, 3*i + 3)
    ax.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
    ax.set_title("Denoised")
    ax.axis('off')

plt.tight_layout()
plt.show()
