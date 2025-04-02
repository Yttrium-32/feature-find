from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.models import Model
import numpy
import cv2

class FeatureExtractor:
    def __init__(self):
        base_model = ResNet50(weights='imagenet')
        self.model = Model(
            inputs=base_model.input,
            outputs=base_model.get_layer('avg_pool').output
        )

    def describe(self, image):
        image_resized = cv2.resize(image, (224, 224))
        image_array = numpy.expand_dims(image_resized, axis=0).astype(numpy.float32)
        image_array = preprocess_input(image_array)
        features = self.model.predict(image_array, verbose=0)
        return features.flatten()

