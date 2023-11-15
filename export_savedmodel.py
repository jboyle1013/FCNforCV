import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras


def export(input_h5_file, export_path):
    # The export path contains the name and the version of the model
    tf.keras.backend.set_learning_phase(0)  # Ignore dropout at inference
    model = tf.keras.models.load_model(input_h5_file)
    model.save(export_path, save_format='tf')
    print(f"SavedModel created at {export_path}")


def predict_on_image(image_path):
    # Load and preprocess the image
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(256, 256))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = image / 255.0  # Normalize the pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add a batch dimension

    # Make predictions using the model
    predictions = model.predict(image)

    # Get the class with the highest probability as the predicted class
    predicted_class = np.argmax(predictions)
    # Assuming you have a list of class names, you can map the class index to a class name
    class_names = ['d10', 'd12', 'd20', 'd4', 'd6', 'd8']  # Update with your actual class names
    predicted_class_name = class_names[predicted_class]

    return predicted_class_name


def testagainstimage():
    image_size = (256, 256)
    testdir = "testdice"
    for images in os.listdir(testdir):
        type = images.split("_")[0]
        img_path = testdir + "/" + images
        img = keras.utils.load_img(
            img_path, target_size=image_size
        )
        plt.imshow(img)

        img_array = keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis

        predictions = model.predict(img_array)
        score = np.argmax(predictions)
        class_names = ['d10', 'd12', 'd20', 'd4', 'd6', 'd8']  # Update with your actual class names
        predicted_class_name = class_names[score]
        print(f"Image Name: {images}")
        print(f"Actual Dice Type: {type}")
        print(f"Predicted Type: {predicted_class_name}")


def testagainstimagewmodel(model):
    image_size = (256, 256)
    testdir = "testdice"
    for images in os.listdir(testdir):
        type = images.split("_")[0]
        img_path = testdir + "/" + images
        img = keras.utils.load_img(
            img_path, target_size=image_size
        )
        plt.imshow(img)

        img_array = keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis

        predictions = model.predict(img_array)
        score = np.argmax(predictions)
        class_names = ['d10', 'd12', 'd20', 'd4', 'd6', 'd8']  # Update with your actual class names
        predicted_class_name = class_names[score]
        print(f"Image Name: {images}")
        print(f"Actual Dice Type: {type}")
        print(f"Predicted Type: {predicted_class_name}")



if __name__ == "__main__":
    input_h5_file = 'DiceFCN.keras'
    model = tf.keras.models.load_model(input_h5_file)
    testagainstimage()
    # export_path = './models/dice_model'
    # export(input_h5_file, export_path)
