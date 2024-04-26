import tensorflow as tf


@tf.function
def toroidal_difference(a, b):
    """
        Computes the toroidal difference between two vectors.
        Args:
            a,    [d,] vector
            b,    [d,] vector
        Returns:
            distance,    [d,] vector of toroidal differences
    """
    return tf.math.floormod(a - b + 0.5, 1.0) - 0.5
    
@tf.function
def toroidal_distance(a, b, axis=None):
    return tf.reduce_sum(toroidal_difference(a, b)**2.0, axis=axis)

@tf.function
def toroidal_pairwise_dist(A, B):
    """
        Computes pairwise distances between each elements of A and each elements of B.
        Args:
            A,    [d,m] matrix
            B,    [d,n] matrix
        Returns:
            distances,    [m,n] matrix of pairwise distances
    """
    toroid_diffs = toroidal_difference(tf.expand_dims(tf.transpose(A), 1), tf.expand_dims(tf.transpose(B), 0))
    return tf.reduce_sum(toroid_diffs**2,axis=2)