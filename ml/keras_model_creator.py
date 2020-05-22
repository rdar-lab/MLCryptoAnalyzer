from abc import ABC, abstractmethod

import numpy as np
from keras import backend
from keras.layers import Dense, Input, Dropout, Activation, Flatten, Conv1D, MaxPooling1D, BatchNormalization, \
    Embedding, LSTM
from keras.models import Model


def _create_binary_embedding():
    """
    Creates an embedding matrix based on the binary representation
    Keras will transform each one byte to a one-hot vector of size 256
    This is required for RNN networks
    :return: An embedding layer
    """
    emb_matrix = np.zeros((256, 256))
    for i in range(0, 256):
        emb_matrix[i, i] = 1

    embedding_layer = Embedding(256, 256, trainable=False)
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])
    return embedding_layer


class _NonTrainableBatchNormalization(BatchNormalization):
    """
    This class fixes an issue with BatchNormalization in Keras
    See:
        https://github.com/keras-team/keras/issues/9522
        https://stackoverflow.com/questions/48230122/keras-batchnormalization-differing-results-in-trainin-and-evaluation-on-trainin
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs, trainable=False, momentum=1.0)

    def call(self, inputs, training=None):
        return super(
            _NonTrainableBatchNormalization, self).call(inputs, training=False)


class BaseKerasModelCreator(ABC):
    """
    The base parent for the BaseKerasModelCreator
    This class is responsible for creating the Keras model (before training or applying pre-trained weights)
    """
    _max_input_record_size: int
    _dropout_rate: float
    _output_categories = int
    _embedded_binary_data: bool

    def __init__(self, output_categories: int, dropout_rate: float = 0.0, max_input_record_size: int = 2000,
                 embedded_binary_data: bool = False):
        """
        Ctor for the BaseKerasModelCreator
        :param output_categories: The num of categories in the output of the model - the size of the softmax result
        :param dropout_rate: Dropout rate to apply
        :param max_input_record_size: The maximum input record size. Can be NULL for RNN otherwise required
        :param embedded_binary_data: If to embedded the input layer (required for RNN, but can be used on others)
        """
        self._max_input_record_size = max_input_record_size
        self._dropout_rate = dropout_rate
        self._output_categories = output_categories
        self._embedded_binary_data = embedded_binary_data

    def _apply_dropout(self, state):
        if self._dropout_rate > 0.0:
            state = Dropout(rate=self._dropout_rate)(state)
        return state

    def _inputs_creator(self, is_in_learning_mode):
        """
        Creates and returns the input layer, and the initial state
        :param is_in_learning_mode: bool indicating if the model to be created should be in learning mode
        :return: A tuple of the input def and the initial state
        """
        if self._embedded_binary_data:
            input_def = Input(shape=(self._max_input_record_size,), dtype='uint8')
            state = _create_binary_embedding()(input_def)
        else:
            input_def = state = Input(shape=(self._max_input_record_size,), dtype='float32')
            if is_in_learning_mode:
                state = BatchNormalization(momentum=0.0)(state)
            else:
                state = _NonTrainableBatchNormalization()(state)

        return input_def, state

    def _final_model_creator(self, inputs, state, is_in_learning_mode):
        """
        Creates the final layers for the model and then finalize the model and compiles it
        :param inputs: The input def
        :param state: The before final state
        :param is_in_learning_mode: bool indicating if the model to be created should be in learning mode
        :return: The Keras model
        """
        # Making sure final outputs are ready for FC layer
        if backend.ndim(state) > 2:
            state = Flatten()(state)

        # FC layer that for categorical output
        state = Dense(units=self._output_categories)(state)
        state = self._apply_dropout(state)

        # Final softmax activation
        state = Activation('softmax')(state)

        model = Model(inputs=inputs, outputs=state, trainable=is_in_learning_mode)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
        return model

    @abstractmethod
    def create_model(self, is_in_learning_mode=True):
        """
        The main method used for creating the model
        :param is_in_learning_mode: bool indicating if the model to be created should be in learning mode
        :return: The Keras model
        """
        raise NotImplementedError()


class KerasRNNModelCreator(BaseKerasModelCreator):
    """
    An RNN model creator.
    An RMM (Recurrent Neural Network) can handle inputs from variable sizes without increasing the trainable parameters
    This model will work as following:
        INPUT -> Embedding -> Conv1D -> LSTM -> ... -> LSTM -> DENSE -> SOFTMAX
    """
    __hidden_layers: int
    __units_per_layer: int

    def __init__(self, output_categories: int, dropout_rate: float = 0.0,
                 hidden_layers: int = 1, units_per_layer: int = 100):
        """
        Ctor for the KerasRNNModelCreator class
        :param output_categories: The num of categories in the output of the model - the size of the softmax result
        :param dropout_rate: Dropout rate to apply
        :param hidden_layers: Number of hidden layers
        :param units_per_layer: Number of units per layer
        """
        super().__init__(output_categories, dropout_rate, max_input_record_size=None, embedded_binary_data=True)
        self.__hidden_layers = hidden_layers
        self.__units_per_layer = units_per_layer

    def create_model(self, is_in_learning_mode=True):
        """
        The main method used for creating the model
        :param is_in_learning_mode: bool indicating if the model to be created should be in learning mode
        :return: The Keras model
        """
        input_def, state = self._inputs_creator(is_in_learning_mode)

        # First add a Conv1D with kernel size of 128 and with a stride of 128
        # This will be processing each 128b of data to a (units_per_layer) shape vector
        state = Conv1D(self.__units_per_layer, 128, strides=128)(state)

        state = self._apply_dropout(state)

        # Hidden layers of the RNN
        for i in range(self.__hidden_layers):
            state = LSTM(self.__units_per_layer, return_sequences=True)(state)
            state = self._apply_dropout(state)

        # Final hidden layer that will return a (units_per_layer) state to be processed by the last softmax layer
        state = LSTM(self.__units_per_layer, return_sequences=False)(state)
        state = self._apply_dropout(state)

        return self._final_model_creator(input_def, state, is_in_learning_mode)


class KerasCNNModelCreator(BaseKerasModelCreator):
    """
    A CNN (Convolutional Neural Network).
    A CMN uses filters (kernels) to apply the same mathematical calculations over moving regions of the data based
    on the kernel and strides.
    The model will work as following:
       INPUT -> CONV1D -> ACTIVATION -> MAX_POOL -> ... -> CONV1D -> ACTIVATION -> MAX_POOL ->  FLATTEN -> DENSE -> SOFTMAX
    """
    __hidden_layers: int
    __filters_in_first_layer: int
    __kernel_size_in_first_layer: int
    __hidden_layers_activation: str
    __stride_in_first_layer: int

    def __init__(self, output_categories: int, dropout_rate: float = 0.0, max_input_record_size: int = 2000,
                 hidden_layers: int = 1, filters_in_first_layer: int = 300, kernel_size_in_first_layer: int = 32,
                 stride_in_first_layer: int = 0,
                 hidden_layers_activation: str = "relu"):
        """
        CTOR for the KerasCNNModelCreator
        :param output_categories:The num of categories in the output of the model - the size of the softmax result
        :param dropout_rate: Dropout rate to apply
        :param max_input_record_size: The maximum input record size.
        :param hidden_layers: Number of hidden layers
        :param filters_in_first_layer: Number of filters in the first layer (doubles every layer)
        :param kernel_size_in_first_layer: Size of the kernel in the first layer (halfs every layer)
        :param stride_in_first_layer: The stride in each layer (halfs every layer)
        :param hidden_layers_activation: The activation for each layer
        """
        super().__init__(output_categories, dropout_rate, max_input_record_size, embedded_binary_data=True)
        self.__hidden_layers = hidden_layers
        self.__filters_in_first_layer = filters_in_first_layer
        self.__kernel_size_in_first_layer = kernel_size_in_first_layer
        self.__hidden_layers_activation = hidden_layers_activation

        if stride_in_first_layer == 0:
            stride_in_first_layer = int(kernel_size_in_first_layer / 10)

        self.__stride_in_first_layer = stride_in_first_layer

    def create_model(self, is_in_learning_mode=True):
        """
        The main method used for creating the model
        :param is_in_learning_mode: bool indicating if the model to be created should be in learning mode
        :return: The Keras model
        """
        input_def, state = self._inputs_creator(is_in_learning_mode)

        # Hidden layers
        for i in range(1, self.__hidden_layers + 1):
            # Every layer we add more filters to gain deeper understanding
            filters = self.__filters_in_first_layer * i

            # Every layer the kernel size should be smaller
            kernel_size = int(self.__kernel_size_in_first_layer / i)

            # Every layer the stride should be smaller
            stride = int(self.__stride_in_first_layer / i)
            if stride < 1:
                stride = 1

            state = Conv1D(filters, kernel_size, strides=stride)(state)
            state = self._apply_dropout(state)

            state = Activation(self.__hidden_layers_activation)(state)
            state = MaxPooling1D(2)(state)

        return self._final_model_creator(input_def, state, is_in_learning_mode)


class KerasFullyConnectedNNModelCreator(BaseKerasModelCreator):
    """
    A regular fully connected Neural Network
    This model will construct a dense FC of n layers
    The model will work as following:
        INPUT -> DENSE -> ... -> DENSE -> DENSE -> SOFTMAX
    """
    __hidden_layers: int
    __units_per_layer: int
    __hidden_layers_activation: str

    def __init__(self, output_categories: int, dropout_rate: float = 0.0, max_input_record_size: int = 2000,
                 embedded_binary_data: bool = False,
                 hidden_layers: int = 1, units_per_layer: int = 128, hidden_layers_activation: str = "relu"):
        """
        Ctor for the KerasFullyConnectedNNModelCreator class
        :param output_categories: The num of categories in the output of the model - the size of the softmax result
        :param dropout_rate: Dropout rate to apply
        :param max_input_record_size: The maximum input record size
        :param embedded_binary_data: Indication if to embedded the input data. If not BatchNormalization will be added norm the input
        :param hidden_layers: Number of hidden layers
        :param units_per_layer: Number of units in each hidden layer
        :param hidden_layers_activation: The activation for the hidden layers
        """
        super().__init__(output_categories, dropout_rate, max_input_record_size, embedded_binary_data)
        self.__hidden_layers = hidden_layers
        self.__units_per_layer = units_per_layer
        self.__hidden_layers_activation = hidden_layers_activation

    def create_model(self, is_in_learning_mode=True):
        """
        The main method used for creating the model
        :param is_in_learning_mode: bool indicating if the model to be created should be in learning mode
        :return: The Keras model
        """
        input_def, state = self._inputs_creator(is_in_learning_mode)

        # Since FCNN is liner - flatten the input first
        if backend.ndim(state) > 2:
            state = Flatten()(state)

        # Hidden layers
        for i in range(1, self.__hidden_layers + 1):
            state = Dense(units=self.__units_per_layer, activation=self.__hidden_layers_activation)(state)
            state = self._apply_dropout(state)

        return self._final_model_creator(input_def, state, is_in_learning_mode)
