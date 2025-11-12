"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
from .nn_basic import Parameter, Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        neg_x = ops.mul_scalar(x, -1.0)
        exp_neg_x = ops.exp(neg_x)
        one_plus_exp = ops.add_scalar(exp_neg_x, 1.0)
        ones = init.ones_like(one_plus_exp)
        return ops.divide(ones, one_plus_exp)
        ### END YOUR SOLUTION

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        
        k = 1.0 / hidden_size
        bound = (k ** 0.5)
        
        weight_ih = init.rand(input_size, hidden_size, low=-bound, high=bound, device=device, dtype=dtype)
        if device is not None:
            weight_ih_ndarray = weight_ih.realize_cached_data()
            weight_ih_ndarray = weight_ih_ndarray.to(device)
            weight_ih = Tensor(weight_ih_ndarray, device=device, dtype=dtype)
        self.W_ih = Parameter(weight_ih, device=device, dtype=dtype)
        
        weight_hh = init.rand(hidden_size, hidden_size, low=-bound, high=bound, device=device, dtype=dtype)
        if device is not None:
            weight_hh_ndarray = weight_hh.realize_cached_data()
            weight_hh_ndarray = weight_hh_ndarray.to(device)
            weight_hh = Tensor(weight_hh_ndarray, device=device, dtype=dtype)
        self.W_hh = Parameter(weight_hh, device=device, dtype=dtype)
        
        if bias:
            bias_ih = init.rand(hidden_size, low=-bound, high=bound, device=device, dtype=dtype)
            if device is not None:
                bias_ih_ndarray = bias_ih.realize_cached_data()
                bias_ih_ndarray = bias_ih_ndarray.to(device)
                bias_ih = Tensor(bias_ih_ndarray, device=device, dtype=dtype)
            self.bias_ih = Parameter(bias_ih, device=device, dtype=dtype)
            
            bias_hh = init.rand(hidden_size, low=-bound, high=bound, device=device, dtype=dtype)
            if device is not None:
                bias_hh_ndarray = bias_hh.realize_cached_data()
                bias_hh_ndarray = bias_hh_ndarray.to(device)
                bias_hh = Tensor(bias_hh_ndarray, device=device, dtype=dtype)
            self.bias_hh = Parameter(bias_hh, device=device, dtype=dtype)
        else:
            self.bias_ih = None
            self.bias_hh = None
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs = X.shape[0]
        
        if h is None:
            h = init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype)
        
        xWih = ops.matmul(X, self.W_ih)
        hWhh = ops.matmul(h, self.W_hh)
        
        if self.bias:
            xWih = xWih + self.bias_ih.reshape((1, self.hidden_size)).broadcast_to(xWih.shape)
            hWhh = hWhh + self.bias_hh.reshape((1, self.hidden_size)).broadcast_to(hWhh.shape)
        
        output = xWih + hWhh
        
        if self.nonlinearity == 'tanh':
            return ops.tanh(output)
        elif self.nonlinearity == 'relu':
            return ops.relu(output)
        else:
            raise ValueError(f"Unsupported nonlinearity: {self.nonlinearity}")
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.nonlinearity = nonlinearity
        
        self.rnn_cells = []
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            cell = RNNCell(layer_input_size, hidden_size, bias=bias, nonlinearity=nonlinearity, device=device, dtype=dtype)
            self.rnn_cells.append(cell)
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, _ = X.shape
        
        if h0 is None:
            h0_splits = [None] * self.num_layers
        else:
            h0_splits = list(ops.split(h0, axis=0))
        
        X_splits = ops.split(X, axis=0)
        layer_inputs = list(X_splits)
        
        h_n_list = []
        
        for layer_idx in range(self.num_layers):
            cell = self.rnn_cells[layer_idx]
            h_layer = h0_splits[layer_idx]
            
            layer_outputs = []
            for t in range(seq_len):
                x_t = layer_inputs[t]
                h_layer = cell(x_t, h_layer)
                layer_outputs.append(h_layer)
            
            layer_inputs = layer_outputs
            h_n_list.append(h_layer)
        
        output = ops.stack(layer_outputs, axis=0)
        h_n = ops.stack(h_n_list, axis=0)
        
        return output, h_n
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        
        k = 1.0 / hidden_size
        bound = (k ** 0.5)
        
        weight_ih = init.rand(input_size, 4 * hidden_size, low=-bound, high=bound, device=device, dtype=dtype)
        if device is not None:
            weight_ih_ndarray = weight_ih.realize_cached_data()
            weight_ih_ndarray = weight_ih_ndarray.to(device)
            weight_ih = Tensor(weight_ih_ndarray, device=device, dtype=dtype)
        self.W_ih = Parameter(weight_ih, device=device, dtype=dtype)
        
        weight_hh = init.rand(hidden_size, 4 * hidden_size, low=-bound, high=bound, device=device, dtype=dtype)
        if device is not None:
            weight_hh_ndarray = weight_hh.realize_cached_data()
            weight_hh_ndarray = weight_hh_ndarray.to(device)
            weight_hh = Tensor(weight_hh_ndarray, device=device, dtype=dtype)
        self.W_hh = Parameter(weight_hh, device=device, dtype=dtype)
        
        if bias:
            bias_ih = init.rand(4 * hidden_size, low=-bound, high=bound, device=device, dtype=dtype)
            if device is not None:
                bias_ih_ndarray = bias_ih.realize_cached_data()
                bias_ih_ndarray = bias_ih_ndarray.to(device)
                bias_ih = Tensor(bias_ih_ndarray, device=device, dtype=dtype)
            self.bias_ih = Parameter(bias_ih, device=device, dtype=dtype)
            
            bias_hh = init.rand(4 * hidden_size, low=-bound, high=bound, device=device, dtype=dtype)
            if device is not None:
                bias_hh_ndarray = bias_hh.realize_cached_data()
                bias_hh_ndarray = bias_hh_ndarray.to(device)
                bias_hh = Tensor(bias_hh_ndarray, device=device, dtype=dtype)
            self.bias_hh = Parameter(bias_hh, device=device, dtype=dtype)
        else:
            self.bias_ih = None
            self.bias_hh = None
        
        self.sigmoid = Sigmoid()
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs = X.shape[0]
        
        if h is None:
            h0 = init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype)
            c0 = init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype)
        else:
            h0, c0 = h
        
        xWih = ops.matmul(X, self.W_ih)
        hWhh = ops.matmul(h0, self.W_hh)
        
        if self.bias:
            xWih = xWih + self.bias_ih.reshape((1, 4 * self.hidden_size)).broadcast_to(xWih.shape)
            hWhh = hWhh + self.bias_hh.reshape((1, 4 * self.hidden_size)).broadcast_to(hWhh.shape)
        
        gates = xWih + hWhh
        
        gates_reshaped = ops.reshape(gates, (bs, 4, self.hidden_size))
        gates_split = ops.split(gates_reshaped, axis=1)
        i_gate = gates_split[0]
        f_gate = gates_split[1]
        g_gate = gates_split[2]
        o_gate = gates_split[3]
        
        i = self.sigmoid(i_gate)
        f = self.sigmoid(f_gate)
        g = ops.tanh(g_gate)
        o = self.sigmoid(o_gate)
        
        c_prime = ops.add(ops.multiply(f, c0), ops.multiply(i, g))
        h_prime = ops.multiply(o, ops.tanh(c_prime))
        
        return (h_prime, c_prime)
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        
        self.lstm_cells = []
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            cell = LSTMCell(layer_input_size, hidden_size, bias=bias, device=device, dtype=dtype)
            self.lstm_cells.append(cell)
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, _ = X.shape
        
        if h is None:
            h0_splits = [None] * self.num_layers
            c0_splits = [None] * self.num_layers
        else:
            h0, c0 = h
            h0_splits = list(ops.split(h0, axis=0))
            c0_splits = list(ops.split(c0, axis=0))
        
        X_splits = ops.split(X, axis=0)
        layer_inputs = list(X_splits)
        
        h_n_list = []
        c_n_list = []
        
        for layer_idx in range(self.num_layers):
            cell = self.lstm_cells[layer_idx]
            h_layer = h0_splits[layer_idx]
            c_layer = c0_splits[layer_idx]
            
            if h_layer is None or c_layer is None:
                hidden_state = None
            else:
                hidden_state = (h_layer, c_layer)
            
            layer_outputs = []
            for t in range(seq_len):
                x_t = layer_inputs[t]
                h_layer, c_layer = cell(x_t, hidden_state)
                hidden_state = (h_layer, c_layer)
                layer_outputs.append(h_layer)
            
            layer_inputs = layer_outputs
            h_n_list.append(h_layer)
            c_n_list.append(c_layer)
        
        output = ops.stack(layer_outputs, axis=0)
        h_n = ops.stack(h_n_list, axis=0)
        c_n = ops.stack(c_n_list, axis=0)
        
        return (output, (h_n, c_n))
        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        weight = init.randn(num_embeddings, embedding_dim, device=device, dtype=dtype)
        self.weight = Parameter(weight)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        l, b = x.shape
        x_one_hot = init.one_hot(self.num_embeddings, x, device=x.device, dtype=x.dtype)
        x_one_hot = ops.reshape(x_one_hot, (l * b, self.num_embeddings))
        output = ops.matmul(x_one_hot, self.weight)
        return ops.reshape(output, (l, b, self.embedding_dim))
        ### END YOUR SOLUTION
