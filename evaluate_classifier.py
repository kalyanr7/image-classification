import tensorflow as tf


def evaluate_singe_image(image_path= "PetImages/Cat/6779.jpg", model_path="saved_model.h5", image_size=(180, 180)):
    
    model = tf.keras.models.load_model(model_path)
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=image_size
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    score = predictions[0]

    return print("This image is %.2f percent cat and %.2f percent dog.", (100 * (1 - score), 100 * score))

if __name__ == '__main__':

    evaluate_singe_image(image_path= "PetImages/Cat/6779.jpg", model_path="saved_model.h5", image_size=(180, 180))
