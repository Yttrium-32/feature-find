from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Model
import numpy
import cv2

class FeatureExtractor:
    def __init__(self):
        self.base_model = ResNet50(weights='imagenet')
        self.model = Model(
            inputs=self.base_model.input,
            outputs=[
                self.base_model.get_layer('avg_pool').output,
                self.base_model.output
            ]
        )

    def describe(self, image):
        image_resized = cv2.resize(image, (224, 224))
        image_array = numpy.expand_dims(image_resized, axis=0).astype(numpy.float32)
        image_array = preprocess_input(image_array)

        features, prediction = self.model.predict(image_array, verbose=0)

        decoded_pred = decode_predictions(prediction, top=1)[0][0]
        label = decoded_pred[1]

        return features.flatten(), label

