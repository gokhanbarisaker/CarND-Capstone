import tensorflow as tf

from keras import backend as K
from keras.models import Sequential, load_model
from tensorflow import Graph

class LightClassifier():
    def __init__(self):
        # Set TF configuration
        config = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2))
        config.gpu_options.allow_growth = True

        self.graph = Graph()
        with self.graph.as_default():
            self.session = tf.Session(config=config)
            with self.session.as_default():

                # Load model
                K.set_learning_phase(0)
                #with open("./data/model.json", 'r') as json_file:
                #    loaded_model_json = json_file.read()
                #model = model_from_json(loaded_model_json)

                model = load_model("./data/model1.h5")
                K.set_learning_phase(0)
                
                # compile requirement for inrefence...
                model.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['acc'])
                K.set_learning_phase(0)

                self.model = model

    def classify(self, input):
        with self.graph.as_default():
            with self.session.as_default():
                result = self.model.predict(input)[0]
        return result