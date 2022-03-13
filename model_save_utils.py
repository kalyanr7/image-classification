import tensorflow as tf


def save_model(model, save_model_path: str):
    """
    Arguments:
    model: pass keras model
    model_name: model name with which the trained model should be saved.

    """

    tf.keras.models.save_model(model, save_model_path + "/" + "final_epoch", save_format='tf')
    tf.keras.models.save_model(model, save_model_path + "/" + "final_epoch.h5", save_format='h5')

    return print("final epoch model is saved at " + save_model_path)
