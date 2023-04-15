"""The Poisson Binomial distribution class."""
import tensorflow.compat.v2 as tf
import numpy as np
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import tensor_util
import pyfftw

class PoissonBinomial(distribution.Distribution):

    def __init__(self,
               probs=None,
               validate_args=False,
               allow_nan_stats=True,
             name='PoissonBinomial'):
        self._dtype = dtype_util.common_dtype([probs], tf.float32)
        self._probs = tensor_util.convert_nonref_to_tensor(
        probs, dtype=self._dtype)
        self._total_count = tf.shape(self._probs)
        self._pmf_list = self.get_pmf()
        self._cdf_list = tf.math.cumsum(self._pmf_list)

        
