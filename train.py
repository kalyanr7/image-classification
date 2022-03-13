import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import argparse
from data_loaders import TfKerasDatasetLoader
from classifier_architectures import ClassifierTfKeras
from model_save_utils import save_model

parser = argparse.ArgumentParser()
parser.add_argument('--train_dataset_dir', type=str, default="./binary_classifier/train",
                    help='path to the train dataset')

parser.add_argument('--validation_dataset_dir', type=str, default="./binary_classifier/train",
                    help='path to the validation dataset')

parser.add_argument('--train_batch_size', type=int, default=16,
                    help='input batch size for training (default: 1)')

parser.add_argument('--epochs', type=int, default=10,
                    help='total number of epochs for model to train (default: 10)')

parser.add_argument('--validation_batch_size', type=int, default=1,
                    help='input batch size for validation (default: 1)')

parser.add_argument('--class_mode', type=str, default="binary",
                    help='class mode (default: "binary")')    

parser.add_argument('--interpolation', type=str, default="lanczos",
                    help='interpolation method for resizing (default: "lanczos")') 

parser.add_argument('--input_size', type=tuple, default=(224,224,3),
                    help='input size to which image to be resized (default: (224,224,3))')  

parser.add_argument('--include_top', type=bool, default=True,
                    help='input batch size for training (default: 64)')

parser.add_argument('--final_dense', type=int, default=1,
                    help='Neurons in final layer (default: 1)')

parser.add_argument("--weights", type=str, default="imagenet", 
                    help="weights for pretrained model,  (default: 10)")

parser.add_argument("--feature_extractor", type=str, default="EfficientNetB0",
                    help="Base feature extractor (default: EfficientNetB0)")

parser.add_argument('--pretrained_model_path', type=str, default="./base_custom_pretrained/final_epoch.h5",
                    help='path to the pretrained base model (default: ./base_custom_pretrained)')

parser.add_argument('--distributed_strategy', type=str, 
                    help='distributed strategy for multi gpu traiing')

parser.add_argument("--pretrained_base", type=str, default="custom-trained-weights", 
                    help="flag to choose custom trained vs imagenet trained weights (default: custom-trained-weights)")

parser.add_argument('--modelcheckpoint_callback_path', type=str, default="./trained_models",
                    help='input batch size for training (default: "./trained_models")')

parser.add_argument('--optimizer', type=str, default="adam",
                    help='optimizer function to be used (default: adam)')

parser.add_argument("--loss", type=str, default="binary_crossentropy",
                    help="loss function to be used (default: binary_crossentropy)")

parser.add_argument("--final_epoch_model_save_path", type=int, default=10, metavar="N",
                    help="number of epochs to train (default: 10)")

args = parser.parse_args()

height, width, channels = args.input_size

data_loader = TfKerasDatasetLoader(train_dataset_dir=args.train_dataset_dir,
        validation_dataset_dir =args.validation_dataset_dir,
        train_batch_size = args.train_batch_size,
        class_mode = args.class_mode,
        interpolation = args.interpolation,
        resize_height = height,
        resize_width = width,
        )

train_data, validation_data = data_loader.build_dataset_flow_from_directory()

ImageClassifier = ClassifierTfKeras( 
                input_shape = (height,width,channels), 
                include_top = False, 
                final_dense = args.final_dense, 
                weights = args.weights, 
                feature_extractor =args.feature_extractor, 
                pretrained_model_path = args.pretrained_model_path, 
                distributed_strategy =  args.distributed_strategy, 
                pretrained_base = args.pretrained_base,
                )

model = ImageClassifier.load_pretrained_base()
# model.summary()

#keras.utils.plot_model(model, show_shapes=True)

callbacks = [
    keras.callbacks.ModelCheckpoint(args.modelcheckpoint_callback_path),
]
model.compile(
    optimizer=args.optimizer,
    loss=args.loss,
    metrics=["accuracy"],
)
model.fit(
    train_data, epochs=args.epochs, callbacks=callbacks, validation_data=validation_data, steps_per_epoch = 1, validation_steps = 1,
)

save_model(model, args.final_epoch_model_save_path)


