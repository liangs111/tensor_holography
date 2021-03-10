import numpy as np
import tensorflow as tf

# interleave tensor without using space_to_depth API, TensorRT compatible
def tf_interleave_nonnative(r, x): 
    if r == 1: 
        return x
    else: 
        batch, depth, height, width = x.shape #NCHW
        reduced_height = height // r
        reduced_width = width // r
        y = tf.reshape(x, [batch, depth, reduced_height, r, reduced_width, r])
        z = tf.reshape(tf.transpose(y, [0,3,5,1,2,4]), (batch, -1, reduced_height, reduced_width))
        return z

# de-interleave tensor without using space_to_depth API, TensorRT compatible
def tf_deinterleave_nonnative(r,x):
    if r == 1:
        return x
    else:  
        batch, depth, height, width = x.shape #NCHW
        expanded_height = height * r
        expanded_width = width * r
        y = tf.reshape(x, [batch, r, r, depth//r//r, height, width])
        z = tf.reshape(tf.transpose(y, [0,3,4,1,5,2]), (batch, -1, expanded_height, expanded_width))
        return z

# initialize weights (adapted from DeepFocus [Xiao et al. 18])
def tf_init_weights(shape, init_method='xavier', xavier_params = (None, None), r = 0.5, seed=0, is_complex=False):
    (fan_in, fan_out) = xavier_params        
    high = np.sqrt(r*2.0/(fan_in+fan_out))
    low = -high
    if is_complex:
        return tf.complex(tf.Variable(tf.random.uniform(shape, minval=low, maxval=high, dtype=tf.float32, seed=seed)),
                        tf.Variable(tf.random.uniform(shape, minval=low, maxval=high, dtype=tf.float32, seed=seed)))
    else:
        return tf.Variable(tf.random.uniform(shape, minval=low, maxval=high, dtype=tf.float32, seed=seed))