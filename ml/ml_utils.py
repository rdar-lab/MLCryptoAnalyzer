import numpy as np
import termplot
from keras import Model

from datastore.encryption_datastore_constants import EncryptionDatastoreConstants
from ml.keras_data_generator import KerasOnlineDataGenerator, BaseKerasDataGenerator
from ml.keras_model_creator import BaseKerasModelCreator


def train_and_evaluate(model_creator: BaseKerasModelCreator, evaluation_field, possible_values, plaintext_generators,
                       encryption_generators, training_rounds=50, max_input_size=2000, inputs_per_round=100):
    """
    Creates, trains, evaluate and returns the model
    :param model_creator: The model creator to use
    :param evaluation_field: The target field to evaluate (the output class field)
    :param possible_values: The possible values for the output
    :param plaintext_generators: The plaintext generators to use
    :param encryption_generators: THe encryption generators to use
    :param training_rounds: The number of training rounds (epocs)
    :param max_input_size: The input size
    :param inputs_per_round: The number of inputs per training round
    :return: The model
    """
    generator = KerasOnlineDataGenerator(max_input_size,
                                         EncryptionDatastoreConstants.CIPHERTEXT,
                                         evaluation_field,
                                         possible_values,
                                         plaintext_generators=plaintext_generators,
                                         encryption_generators=encryption_generators,
                                         batch_size=inputs_per_round
                                         )

    model = create_and_train(model_creator, generator, training_rounds)
    evaluation_data_x, evaluation_data_y = generator[0]
    evaluate_model(model, evaluation_data_x, evaluation_data_y)

    return model


def create_and_train(model_creator: BaseKerasModelCreator, generator: BaseKerasDataGenerator, training_rounds=50):
    """
    Creates a model, trains it and returns the result
    :param model_creator: The model creator to use
    :param generator: A data generator for the training
    :param training_rounds: The number of training rounds (epocs)
    :return: The model
    """
    model: Model = model_creator.create_model(is_in_learning_mode=True)
    print("Model Summary:")
    model.summary()

    print("Training:")
    history = model.fit(generator, epochs=training_rounds, verbose=1, use_multiprocessing=False)
    plot_fit_history(history)

    final_model: Model = model_creator.create_model(is_in_learning_mode=False)
    final_model.set_weights(model.get_weights())

    return final_model


def evaluate_model(model, evaluation_data_x, evaluation_data_y):
    """
    Evaluates the performance of the model, and prints the result to the console
    :param model: The model to evaluate
    :param evaluation_data_x: The X data
    :param evaluation_data_y: The Y data
    """
    print("Evaluating:...")
    print("*** Model result *** ")
    scores = model.evaluate(evaluation_data_x, evaluation_data_y, verbose=0)

    accuracy = None

    for metric_index in range(len(model.metrics_names)):
        print(model.metrics_names[metric_index] + " - " + str(scores[metric_index]))

        if model.metrics_names[metric_index] == 'categorical_accuracy':
            accuracy = scores[metric_index]

    if accuracy < 0.7:
        print("Model failure to predict :-(")
    elif accuracy < 0.85:
        print("Model has average performance :-")
    elif accuracy < 0.95:
        print("Model has good performance :-) ")
    else:
        print("Model has excellent performance! ;-)")


def plot_fit_history(history):
    """
    Plots the training history
    :param history: The history lists (from the fit output)
    """
    print()
    print(" **************** ")
    print(" ***   Loss   *** ")
    print(" **************** ")

    loss = history.history['loss']

    for index in range(len(loss)):
        loss[index] = int(loss[index] * 1000)

    termplot.plot(np.asarray(loss).transpose())

    print()
    print(" **************** ")
    print(" *** Accuracy *** ")
    print(" **************** ")

    accuracy = history.history['categorical_accuracy']

    for index in range(len(accuracy)):
        accuracy[index] = int(accuracy[index] * 1000)

    termplot.plot(np.asarray(accuracy).transpose())
