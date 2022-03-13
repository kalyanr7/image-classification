from tensorflow import keras
import tensorflow as tf

class ClassifierTfKeras:

    """Builds tf.keras.Model using provided configs.

    This class contains :

    1. load_imagenet_pretrained() -  this method loads imagenet trained base feature extractor for classification.
    2. load_custom_pretrained() -  this method loads custom trained base feature extractor for classification.
    3. load_pretrained_base() - chooses between load_imagenet_pretrained() and load_custom_pretrained() with multi gpu support based on user defined args

    Usage : 

    "
    ImageClassifier = ClassifierTfKeras( 
                input_shape = (224,224,3), 
                include_top = False, 
                final_dense = 1, 
                weights = "imagenet", 
                feature_extractor ="EfficientNetB0", 
                pretrained_model_path = "./pretrained", 
                distributed_strategy =  None, 
                pretrained_base = "imagenet-trained-weights",
                )
    
    model = ImageClassifier.load_pretrained_base()
    "
    In the above mentioned example, efficientnetb0 architecture with input shape (224,224,3) is loaded with 1 neuron in final layer.

    Arguments:
        input_shape: shape of the input image (h,w,c) - default -(224,224,3)
        include_top: Argument to decide if imagenet 1000 class final layer to be included or not - default : False
        weights: base weights of the imagenet pretrained model
        feature_extractor: Name of the imagnet trained feature extractor to be used - default : "EfficientNetB0"
        distributed_strategy: strategy for distributed training. for eg., MirroredStrategy - if not mentioned, defaults to None
        
        pretrained_model_path: path for the custom pretrained base model - default : "imagenet"
        pretrained_base: choose between "imagenet-trained-weights" and "custom-trained-weights". choosing "custom-trained-weights"
                         arguement will override feature_extractor, final_dense, include_top
    """

    def __init__(self, 
        input_shape = (224,224,3), 
        include_top = False, 
        final_dense = 1, 
        weights = "imagenet", 
        feature_extractor ="EfficientNetB0", 
        pretrained_model_path = None, 
        distributed_strategy =  None, 
        pretrained_base = None,
    ):

        self.input_shape = input_shape
        self.include_top = include_top
        self.final_dense = final_dense
        self.weights     = weights
        self.feature_extractor = feature_extractor
        self.pretrained_model_path = pretrained_model_path
        self.distributed_strategy = distributed_strategy
        self.pretrained_base = pretrained_base

    def load_imagenet_pretrained(self):

        input_layer = keras.layers.Input(shape=self.input_shape)
        eff_net = keras.applications.__dict__[self.feature_extractor](weights=self.weights,include_top = self.include_top,input_tensor = input_layer)
        if self.include_top == True:
            model = eff_net
            print("This is a 1000 class classifier built on top of {} feature extractor with imagenet trained base weights".format(self.feature_extractor))
        else:
            if self.final_dense > 1 :
                global_avg = keras.layers.GlobalAveragePooling2D()(eff_net.output)
                dense_1 = keras.layers.Dense(self.final_dense,activation = 'softmax')(global_avg)
                model = keras.models.Model(inputs=eff_net.inputs,outputs=dense_1)
                print("This is a {} class classifier built on top of {} feature extractor with imagenet trained base weights".format(self.final_dense,self.feature_extractor))

            else:
                global_avg = keras.layers.GlobalAveragePooling2D()(eff_net.output)
                dense_1 = keras.layers.Dense(self.final_dense,activation = 'sigmoid')(global_avg)
                model = keras.models.Model(inputs=eff_net.inputs,outputs=dense_1)
                print("This is a binary classifier built on top of {} feature extractor with imagenet trained base weights".format(self.feature_extractor))
        return model

    def load_custom_pretrained(self):

        model = keras.models.load_model(self.pretrained_model_path)

        return model

    def load_pretrained_base(self):

        if self.pretrained_base == "imagenet-trained-weights":
            print("Imagenet trained base weights are loaded..!!")    
            if self.distributed_strategy is not None:
                strategy = tf.distribute.__dict__[self.distributed_strategy]()
                with strategy.scope():
                    model = self.load_imagenet_pretrained()
            else:
                model = self.load_imagenet_pretrained()

        elif self.pretrained_base == "custom-trained-weights":
            print("Custom trained base weights are loaded..!!")
            if self.distributed_strategy is not None:
                strategy = tf.distribute.__dict__[self.distributed_strategy]()
                with strategy.scope():
                    model = self.load_custom_pretrained()
            else:
                model = self.load_custom_pretrained()

        return model




if __name__ == '__main__':

    ImageClassifier = ClassifierTfKeras(input_shape = (224,224,3),feature_extractor = "ResNet101V2", weights="imagenet", include_top = False, final_dense = 1)
    
    ImageClassifier = ClassifierTfKeras( 
                    input_shape = (224,224,3), 
                    include_top = False, 
                    final_dense = 1, 
                    weights = "imagenet", 
                    feature_extractor ="EfficientNetB0", 
                    pretrained_model_path = "./pretrained", 
                    distributed_strategy =  None, 
                    pretrained_base = "imagenet-trained-weights",
                    )
    
    model = ImageClassifier.load_pretrained_base()
    model.summary()

