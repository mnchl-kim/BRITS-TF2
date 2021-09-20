import tensorflow as tf
from tensorflow.kears import Model
from tensorflow.keras.layers import Layer, Dense, LSTM
from tensorflow.keras.constraints import Constraint


class DiagonalConstraint(Constraint):
    def __init__(self, **kwargs):
        super(DiagonalConstraint, self).__init__(**kwargs)

    def __call__(self, w):
        dim = w.shape[0]
        m = tf.eye(dim)
        w = w * m
        return w


class ZeroDiagonalConstraint(Constraint):
    def __init__(self, **kwargs):
        super(ZeroDiagonalConstraint, self).__init__(**kwargs)

    def __call__(self, w):
        assert w.shape[0] == w.shape[1]

        dim = w.shape[0]
        m = tf.ones((dim, dim)) - tf.eye(dim)
        w = w * m
        return w


class TemporalDecay(Layer):
    def __init__(self, units, diag=False, **kwargs):
        super(TemporalDecay, self).__init__(**kwargs)

        self.units = units
        self.diag = diag

    def build(self, input_shape):
        if self.diag:
            assert self.units == input_shape[-1]
            self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer='he_uniform', trainable=True,
                                     constraint=DiagonalConstraint(), name='kernel')
        else:
            self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer='he_uniform', trainable=True,
                                     name='kernel')

        self.b = self.add_weight(shape=self.units, initializer='zeros', trainable=True, name='bias')

    @tf.function
    def call(self, inputs):
        gamma = tf.nn.relu(tf.matmul(inputs, self.w) + self.b)
        gamma = tf.math.exp(-gamma)
        return gamma


class RITS(Layer):
    def __init__(self, hid_dim, dropout_rate, go_backwards=False, name='RITS', **kwargs):
        super(RITS, self).__init__(name=name, **kwargs)

        self.hid_dim = hid_dim
        self.dropout_rate = dropout_rate
        self.go_backwards = go_backwards

    def build(self, input_shape):
        self.seq_len = input_shape[2]
        self.var_dim = input_shape[3]

        self.rnn_cell = LSTM(units=self.hid_dim, recurrent_dropout=self.dropout_rate, return_state=True,
                             name=self.name + '/lstm')

        self.history_estimation = Dense(units=self.var_dim, activation=None, name=self.name + '/history_estimation')
        self.feature_estimation = Dense(units=self.var_dim, activation=None, name=self.name + '/feature_estimation',
                                        kernel_constraint=ZeroDiagonalConstraint())

        self.temp_decay_x = TemporalDecay(units=self.var_dim, diag=True, name=self.name + '/temp_decay_x')
        self.temp_decay_h = TemporalDecay(units=self.hid_dim, diag=False, name=self.name + '/temp_decay_h')

        self.beta = Dense(units=self.var_dim, activation='sigmoid', name=self.name + '/beta')

        self.out = Dense(units=1, activation='sigmoid', name=self.name + '/out')

    @tf.function
    def call(self, inputs):
        """
        Arguments:
            inputs -- shape: (N, (x, m, d), t, d)
        Returns:
            imputations -- shape: (N, (x_hat, z_hat, c_hat), t, d)
            predictions -- shape: (N, 1)
        """
        
        batch_size = tf.shape(inputs)[0]
        
        # LSTM state
        state_h = tf.zeros(shape=(batch_size, self.hid_dim), dtype=tf.float32)
        state_c = tf.zeros(shape=(batch_size, self.hid_dim), dtype=tf.float32)

        imputations_x = []
        imputations_z = []
        imputations_c = []

        for t in range(self.seq_len):
            if not self.go_backwards:
                x = inputs[:, 0, t, :]
                m = inputs[:, 1, t, :]
                d = inputs[:, 2, t, :]
            else:
                x = inputs[:, 0, self.seq_len - t - 1, :]
                m = inputs[:, 1, self.seq_len - t - 1, :]
                d = inputs[:, 2, self.seq_len - t - 1, :]

            x_hat = self.history_estimation(state_h)
            x_c = m * x + (1 - m) * x_hat

            z_hat = self.feature_estimation(x_c)
            gamma_x = self.temp_decay_x(d)
            beta = self.beta(tf.concat([gamma_x, m], axis=1))
            c_hat = beta * z_hat + (1 - beta) * x_hat
            c_c = m * x + (1 - m) * c_hat

            imputations_x.append(x_hat)
            imputations_z.append(z_hat)
            imputations_c.append(c_hat)

            inputs_cell = tf.concat([c_c, m], axis=1)

            gamma_h = self.temp_decay_h(d)
            state_h = state_h * gamma_h

            _, state_h, state_c = self.rnn_cell(inputs=tf.expand_dims(inputs_cell, axis=1),
                                                initial_state=[state_h, state_c])

        if not self.go_backwards:
            imputations_x = tf.concat([tf.expand_dims(imputation, axis=1) for imputation in imputations_x], axis=1)
            imputations_z = tf.concat([tf.expand_dims(imputation, axis=1) for imputation in imputations_z], axis=1)
            imputations_c = tf.concat([tf.expand_dims(imputation, axis=1) for imputation in imputations_c], axis=1)
        else:
            imputations_x = tf.concat([tf.expand_dims(imputation, axis=1) for imputation in reversed(imputations_x)], axis=1)
            imputations_z = tf.concat([tf.expand_dims(imputation, axis=1) for imputation in reversed(imputations_z)], axis=1)
            imputations_c = tf.concat([tf.expand_dims(imputation, axis=1) for imputation in reversed(imputations_c)], axis=1)

        imputations = tf.concat([tf.expand_dims(imputations_x, axis=1),
                                 tf.expand_dims(imputations_z, axis=1),
                                 tf.expand_dims(imputations_c, axis=1)], axis=1)

        predictions = self.out(state_h)

        return imputations, predictions


class BRITS(Model):
    def __init__(self, hid_dim, dropout_rate, name='BRITS', **kwargs):
        super(BRITS, self).__init__(name=name, **kwargs)

        self.hid_dim = hid_dim
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.rits_forward = RITS(self.hid_dim, self.dropout_rate, go_backwards=False, name="RITS_forward")
        self.rits_backward = RITS(self.hid_dim, self.dropout_rate, go_backwards=True, name="RITS_backward")

    @tf.function
    def call(self, inputs):
        """
        Arguments:
            inputs -- shape: (N, (x, m, d), t, d)
            inputs:
                x -- normalized by standard gaussian distribution N(0, 1)
                m -- observed 1, missed 0
                d -- time delta
        Returns:
            imputations -- shape: (N, (f, b), (x_hat, z_hat, c_hat), t, d)
            # predictions -- shape: (N, (f, b), 1)
            predictions -- shape: (N, 1)
        """

        imputations_forward, predictions_forward = self.rits_forward(inputs)
        imputations_backward, predictions_backward = self.rits_backward(inputs)

        imputations = tf.concat([tf.expand_dims(imputations_forward, axis=1),
                                 tf.expand_dims(imputations_backward, axis=1)], axis=1)
        # predictions = tf.concat([tf.expand_dims(predictions_forward, axis=1),
        #                          tf.expand_dims(predictions_backward, axis=1)], axis=1)
        predictions = (predictions_forward + predictions_backward) / 2

        return imputations, predictions
