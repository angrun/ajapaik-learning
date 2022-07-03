import numpy as np
from app.algorithms import transfer_learning
from app.singleton import singleton
from tensorflow.keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image


@singleton
class TrainigService(object):


    def __init__(self, x=0):
        self.model = None
        # self.image_url = '/home/anna/'
        self.image_url = '/Users/annagrund/PycharmProjects/ajapaik-learning/test_images/'

    def train(self):
        if self.model == None:
            self.model = transfer_learning.get_model()
            print("Training is completed")

    def predict(self, image_url):
        print("I am in predict")
        print(self.model)
        print(image_url)
        if self.model == None:
            return "Model is training"

        try:
            img = image.load_img(f'{image_url}',
                                 target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_batch = np.expand_dims(img_array, axis=0)
            img_preprocessed = preprocess_input(img_batch)
            predictions = self.model.predict(img_preprocessed)

            print(predictions)
            return predictions
        except Exception as e:
            print(e)
            return f"Something went wrong {image_url}"
