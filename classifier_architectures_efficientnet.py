from tensorflow import keras
import efficientnet.tfkeras

class ClassifierEfficientNetTfKeras:

    def __init__(self, input_shape, include_top, final_dense, weights, feature_extractor, pretrained_model_path):

        self.input_shape = input_shape
        self.include_top = include_top
        self.final_dense = final_dense
        self.weights     = weights
        self.feature_extractor = feature_extractor
        self.pretrained_model_path = pretrained_model_path

    def classifier(self):

        input_layer = keras.layers.Input(shape=self.input_shape)
        eff_net = efficientnet.tfkeras.__dict__[self.feature_extractor](weights=self.weights,include_top = self.include_top,input_tensor = input_layer)

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

    def load_pretrained(self): 

        model = keras.models.load_model(self.pretrained_model_path) 

        return model

if __name__ == '__main__':

    efficientnetv1 = ClassifierEfficientNetTfKeras(input_shape = (224,224,3),feature_extractor = "EfficientNetB7", weights="imagenet", include_top = False, final_dense = 1)
    model = efficientnetv1.classifier()
    model.summary()
