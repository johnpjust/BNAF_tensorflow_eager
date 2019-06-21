import tensorflow as tf
import math
import numpy as np

class Sequential(tf.keras.models.Sequential):
    """
    Class that extends ``torch.nn.Sequential`` for computing the output of
    the function alongside with the log-det-Jacobian of such transformation.
    """
    
    # def call(self, inputs: tf.Tensor):
    def call(self, inputs, training=None, mask=None):
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            The input tensor.
        Returns
        -------
        The output tensor and the log-det-Jacobian of this transformation.
        """
        
        log_det_jacobian = 0.
        # for i, module in enumerate(self._modules.values()):
        for i, layer in enumerate(self.layers):
            inputs, log_det_jacobian_ = layer(inputs)
            log_det_jacobian = log_det_jacobian + log_det_jacobian_
        return inputs, log_det_jacobian


class BNAF(tf.keras.models.Sequential):
    """
    Class that extends ``torch.nn.Sequential`` for constructing a Block Neural 
    Normalizing Flow.
    """

    def __init__(self, layers=None, name=None, res: str = None):
    # def __init__(self, *args, res: str = None):
        """
        Parameters
        ----------
        *args : ``Iterable[torch.nn.Module]``, required.
            The modules to use.
        res : ``str``, optional (default = None).
            Which kind of residual connection to use. ``res = None`` is no residual 
            connection, ``res = 'normal'`` is ``x + f(x)`` and ``res = 'gated'`` is
            ``a * x + (1 - a) * f(x)`` where ``a`` is a learnable parameter.
        """
        
        # super(BNAF, self).__init__(*args)
        super(BNAF, self).__init__(name=name)
        self.supports_masking = True
        self._build_input_shape = None
        self._compute_output_and_mask_jointly = True

        # Add to the model any layers passed to the constructor.
        if layers:
          for layer in layers:
            self.add(layer)

        self.res = res
        
        if res == 'gated':
            self.gate = tf.get_variable(name='gate', shape=1, initializer=tf.initializers.random_normal)
            # self.gate = torch.nn.Parameter(torch.nn.init.normal_(torch.Tensor(1)))
    
    # def forward(self, inputs : tf.Tensor):
    # def call(self, inputs: tf.Tensor):
    def call(self, inputs, training=None, mask=None):

        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            The input tensor.
        Returns
        -------
        The output tensor and the log-det-Jacobian of this transformation.
        """

        outputs = inputs
        grad = None

        ### apply layers in TF and get gradients...module in pytorch == layers in tf
        # for module in self._modules.values(): #pytorch implementation
        for layer in self.layers:
            outputs, grad = layer(outputs, grad) #not sure if use "layer" or "layer.call"
            grad = grad if len(grad.shape) == 4 else tf.reshape(grad, (grad.shape + [1, 1]))

        # return outputs, grad ## debug

        assert inputs.shape[-1] == outputs.shape[-1]
        
        if self.res == 'normal':
            return inputs + outputs, tf.reduce_sum(tf.keras.activations.softplus(tf.squeeze(grad)), axis=-1)
        elif self.res == 'gated':
            return tf.nn.sigmoid(self.gate) * outputs + (1 - tf.nn.sigmoid(self.gate)) * inputs, \
                tf.reduce_sum(tf.nn.softplus(tf.squeeze(grad) + self.gate) - \
                 tf.nn.softplus(self.gate), axis=-1)
        else:
            return outputs, tf.reduce_sum(tf.squeeze(grad), axis=-1)
        
    def _get_name(self):
        return 'BNAF(res={})'.format(self.res)
        
        
# class Permutation(torch.nn.Module):
class Permutation(tf.keras.layers.Layer):
    """
    Module that outputs a permutation of its input.
    """
    
    def __init__(self, in_features : int, p : list = None):
        """
        Parameters
        ----------
        in_features : ``int``, required.
            The number of input features.
        p : ``list`` or ``str``, optional (default = None)
            The list of indices that indicate the permutation. When ``p`` is not a
            list, if ``p = 'flip'``the tensor is reversed, if ``p = None`` a random 
            permutation is applied.
        """
        
        super(Permutation, self).__init__()
        
        self.in_features = in_features
        
        if p is None:
            self.p = np.random.permutation(in_features)
        elif p == 'flip':
            self.p = list(reversed(range(in_features)))
        else:
            self.p = p
        
    def call(self, inputs : tf.Tensor, **kwargs):
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            The input tensor.
        Returns
        -------
        The permuted tensor and the log-det-Jacobian of this permutation.
        """
        
        return inputs[:,self.p], 0
    
    def __repr__(self):
        return 'Permutation(in_features={}, p={})'.format(self.in_features, self.p)
    
    
# class MaskedWeight(torch.nn.Module):
class MaskedWeight(tf.keras.layers.Layer):
    """
    Module that implements a linear layer with block matrices with positive diagonal blocks.
    Moreover, it uses Weight Normalization (https://arxiv.org/abs/1602.07868) for stability.
    """
    
    def __init__(self, in_features : int, out_features : int, dim : int, bias : bool = True):
        """
        Parameters
        ----------
        in_features : ``int``, required.
            The number of input features per each dimension ``dim``.
        out_features : ``int``, required.
            The number of output features per each dimension ``dim``.
        dim : ``int``, required.
            The number of dimensions of the input of the flow.
        bias : ``bool``, optional (default = True).
            Whether to add a parametrizable bias.
        """

        super(MaskedWeight, self).__init__()
        self.in_features, self.out_features, self.dim = in_features, out_features, dim

        weight = np.zeros((out_features, in_features))
        # for i in range(dim):
        #     weight[i * out_features // dim:(i + 1) * out_features // dim,
        #            0:(i + 1) * in_features // dim] = torch.nn.init.xavier_uniform_(
        #         torch.Tensor(out_features // dim, (i + 1) * in_features // dim))

        for i in range(dim):
            weight[(i * out_features // dim):((i + 1) * out_features // dim), 0:((i + 1) * in_features // dim)] = \
                tf.get_variable("w", shape=[out_features // dim, (i + 1) * in_features // dim], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, trainable=False).numpy()

        with tf.variable_scope("params", reuse=False):
            self._weight = tf.get_variable("off_diagonal", initializer=tf.cast(weight, dtype=tf.float32), dtype=tf.float32)
        
            # self._weight = torch.nn.Parameter(weight)
            # self._diag_weight = torch.nn.Parameter(torch.nn.init.uniform_(torch.Tensor(out_features, 1)).log())
            self._diag_weight = tf.log(tf.get_variable("diag", shape=(out_features, 1), initializer=tf.initializers.random_uniform(), dtype=tf.float32)) #maybe takes log because we're going to take exp later?

            # self.bias = torch.nn.Parameter(
            #     torch.nn.init.uniform_(torch.Tensor(out_features),
            #                            -1 / math.sqrt(out_features),
            #                            1 / math.sqrt(out_features))) if bias else 0

            self.bias = tf.get_variable("bias", shape=out_features, initializer=tf.initializers.random_uniform(-1 / math.sqrt(out_features), 1 / math.sqrt(out_features))) if bias else 0

        mask_d = np.zeros_like(weight)
        for i in range(dim):
            mask_d[i * (out_features // dim):(i + 1) * (out_features // dim),
                   i * (in_features // dim):(i + 1) * (in_features // dim)] = 1

        # self.register_buffer('mask_d', mask_d)
        self.mask_d = tf.constant(name='mask_d', value=mask_d, dtype=tf.float32)

        mask_o = np.ones_like(weight)
        for i in range(dim):
            mask_o[i * (out_features // dim):(i + 1) * (out_features // dim),
                   i * (in_features // dim):] = 0
            
        # self.register_buffer('mask_o', mask_o)
        self.mask_o = tf.constant(name='mask_o', value=mask_o, dtype=tf.float32)

    def get_weights(self):
        """
        Computes the weight matrix using masks and weight normalization.
        It also compute the log diagonal blocks of it.
        """

        # error in original here i think -- should be self._diag_weight or w_squared_norm is not correct
        w = tf.multiply(tf.exp(self._weight), self.mask_d) + tf.multiply(self._weight, self.mask_o)
        # w = tf.multiply(tf.exp(self._diag_weight), self.mask_d) + tf.multiply(self._weight, self.mask_o)

        w_squared_norm = tf.reduce_sum(tf.math.square(w), axis=-1, keepdims=True)
        
        w = tf.exp(self._diag_weight) * w / tf.sqrt(w_squared_norm)

        ## this piece feeds the log-determinant of the jacobian
        wpl = self._diag_weight + self._weight - 0.5 * tf.log(w_squared_norm)

        # return tf.transpose(w), tf.transpose(wpl)[self.mask_d.byte().t()].view(
        #     self.dim, self.in_features // self.dim, self.out_features // self.dim)

        return tf.transpose(w), tf.reshape(tf.boolean_mask(tf.transpose(wpl),tf.transpose(tf.cast(self.mask_d, tf.bool))),(
            self.dim, self.in_features // self.dim, self.out_features // self.dim))


    # def forward(self, inputs, grad : torch.Tensor = None):
    def call(self, inputs: tf.Tensor, grad: tf.Tensor = None):
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            The input tensor.
        grad : ``torch.Tensor``, optional (default = None).
            The log diagonal block of the partial Jacobian of previous transformations.
        Returns
        -------
        The output tensor and the log diagonal blocks of the partial log-Jacobian of previous 
        transformations combined with this transformation.
        """
        
        w, wpl = self.get_weights()
        
        # g = wpl.transpose(-2, -1).unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1)
        grad_perm = list(range(len(wpl.shape)))
        grad_perm[-1] = len(grad_perm)-2
        grad_perm[-2] = len(grad_perm)-1
        g = tf.tile(tf.expand_dims(tf.transpose(wpl, perm=grad_perm), axis=0), (inputs.shape[0], 1, 1, 1))
        
        # return inputs.matmul(w) + self.bias, torch.logsumexp(
        #     g.unsqueeze(-2) + grad.transpose(-2, -1).unsqueeze(-3), -1) if grad is not None else g

        if grad is not None:
            grad_perm = list(range(len(grad.shape)))
            grad_perm[-1] = len(grad_perm)-2
            grad_perm[-2] = len(grad_perm)-1

        return tf.matmul(inputs, w) + self.bias, tf.reduce_logsumexp(
            tf.expand_dims(g, axis=-2) + tf.expand_dims(tf.transpose(grad, perm=grad_perm), axis=-3), -1) if grad is not None else g

    def __repr__(self):
        return 'MaskedWeight(in_features={}, out_features={}, dim={}, bias={})'.format(
            self.in_features, self.out_features, self.dim, not isinstance(self.bias, int))

    
class Tanh(tf.keras.layers.Layer):
    """
    Class that extends ``torch.nn.Tanh`` additionally computing the log diagonal
    blocks of the Jacobian.
    """

    def call(self, inputs, grad : tf.Tensor = None):
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            The input tensor.
        grad : ``torch.Tensor``, optional (default = None).
            The log diagonal blocks of the partial Jacobian of previous transformations.
        Returns
        -------
        The output tensor and the log diagonal blocks of the partial log-Jacobian of previous 
        transformations combined with this transformation.
        """
        
        # g = - 2 * (inputs - tf.math.log(2) + tf.keras.activations.softplus(- 2 * inputs))
        # return tf.tanh(inputs), (g.view(grad.shape) + grad) if grad is not None else g

        g = - 2 * (inputs - tf.math.log(2.) + tf.keras.activations.softplus(- 2. * inputs))
        return tf.tanh(inputs), (tf.reshape(g,grad.shape) + grad) if grad is not None else g
    