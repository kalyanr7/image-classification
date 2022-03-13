import tensorflow as tf
import os
from glob import glob
import random

class TfKerasDatasetLoader:
    """
    TfKerasDatasetLoader class offers two ways to load the dataset :

    1. ImageDataGenerator.flow_from_directory() which returns data in generator form. To use it, Call build_dataset_flow_from_directory()
    2. tf.data .To use it, Call build_dataset_tfdata()

    Provide the dataset in the below form

    sample_keras_dataset/
    ├── train_data
    │   ├── class1
    │   │   ├── image1.png
    │   │   └── image2.png
    │   └── class2
    │        ├── image1.png
    │        └── image2.png
    └── validation_data
        ├── class1
        │   ├── image1.png
        │   └── image2.png
        └── class2
            ├── image1.png
            └── image2.png

    Arguments:
        train_dataset_dir: the path to the directory containing the  training dataset.
        validation_dataset_dir: the path to the directory containing the  validation dataset.
        train_batch_size: the size of the training batch.
        validation_batch_size: the size of the validation batch.
        class_mode: 'binary' if single neuron in output layer, else, 'categorical'
        interpolation: resize function to use.
        resize_height: height to which image should be resized
        resize_width: width to which image should be resized
    """

    def __init__(
        self,
        train_dataset_dir: str,
        validation_dataset_dir: str,
        train_batch_size: int,
        class_mode: str,
        interpolation: str = 'lanczos',
        resize_height: int = 224,
        resize_width: int = 224,
        validation_batch_size: int = 1,
    ):
        self.train_dataset_dir = train_dataset_dir
        self.validation_dataset_dir = validation_dataset_dir
        self.train_batch_size = train_batch_size
        self.class_mode = class_mode
        self.interpolation = interpolation
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.validation_batch_size = validation_batch_size

    def build_dataset_flow_from_directory(self):
        """Builds and returns the dataset in generator form.

        Loads the dataset and builds the dataset by applying user defined
        operations such as preprocessing/augmenting the data, batching,
        etc.
        """
        data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1. / 255)

        train_data = data_generator.flow_from_directory(
            self.train_dataset_dir,
            target_size=(self.resize_height, self.resize_width),
            batch_size=self.train_batch_size,
            class_mode=self.class_mode,
            interpolation=self.interpolation)

        validation_data = data_generator.flow_from_directory(
            self.validation_dataset_dir,
            target_size=(self.resize_height, self.resize_width),
            batch_size=1,
            class_mode=self.class_mode,
            interpolation=self.interpolation)

        return train_data, validation_data


    def preprocess_image(self, filename):

        print("*******************************", filename)
        image = tf.io.read_file(filename)
        print("*******************************", image.shape)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = image / 255.0
        image = tf.image.resize(image, [224, 224])
        return image
        
    def configure_for_performance(self, dataset, batch_size):
        """
        examples.prefetch(2) will prefetch two elements (2 examples),
        while examples.batch(20).prefetch(2) will prefetch 2 elements
        (2 batches, of 20 examples each)

        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        """

        dataset = dataset.cache('/tmp/dump.tfcache')
        dataset = dataset.shuffle(buffer_size=100)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset
  
    def load_data_from_folder(self, data_dir, batch_size):
        classes = os.listdir(data_dir)
        filenames = glob(data_dir + '/*/*')
        random.shuffle(filenames)
        labels = [classes.index(name.split('/')[-2]) for name in filenames]

        print("************************* filenames", filenames)
        print("************************* cla", classes)
        print("************************* la", labels)

        image_data = tf.data.Dataset.from_tensor_slices(filenames)

        preprocessed_image_data = image_data.map(self.preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        labels = tf.data.Dataset.from_tensor_slices(labels)
        image_label_dataset = tf.data.Dataset.zip((preprocessed_image_data, labels))
        image_label_dataset = self.configure_for_performance(image_label_dataset, batch_size)

        return image_label_dataset

    def build_dataset_tfdata(self):

        train_data = self.load_data_from_folder(self.train_dataset_dir, self.train_batch_size)
        validation_data = self.load_data_from_folder(self.validation_dataset_dir, self.validation_batch_size)

        return train_data, validation_data