import keras
import tensorflow as tf
from model import CNN_model
from generator import Generator
import os
import matplotlib.pyplot as plt
from export_savedmodel import export, testagainstimagewmodel

tf.config.run_functions_eagerly(True)
from newmodel import NewFCN_model


def train(model, train_generator, val_generator, epochs=50):




    checkpoint_path = './snapshots'
    os.makedirs(checkpoint_path, exist_ok=True)
    check_path = checkpoint_path + '/' + 'model_epoch_{epoch:02d}.keras'
    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2),
        tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/model.{epoch:02d}-{val_loss:.2f}.keras'),
        tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    ]
    history = model.fit(train_generator,
                        steps_per_epoch=len(train_generator),
                        epochs=epochs,
                        callbacks=my_callbacks,
                        validation_data=val_generator,
                        validation_steps=len(val_generator),
                        shuffle=True)
    return history, model


if __name__ == "__main__":

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Restrict TensorFlow to only allocate a fraction of GPU memory as needed
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    # Create FCN model


    # The below folders are created using utils.py
    train_dir = 'archive/iam_words/'



    # If you get out of memory error try reducing the batch size
    BATCH_SIZE = 8
    train_generator = Generator(train_dir, BATCH_SIZE, shuffle_images=True, subset="training")
    val_generator = Generator(train_dir, BATCH_SIZE, shuffle_images=True, subset="validation")

    model = CNN_model(train_generator.max_label_len)

    EPOCHS = 50
    history, model = train(model, train_generator, val_generator, epochs=EPOCHS)

    # Accuracy
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    # loss
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    model.save("DiceFCN.keras")
    model.save_weights(
        "DiceFCNWeights", overwrite=True, save_format=None, options=None
    )
    testagainstimagewmodel(model)
    export("DiceFCN.keras", "DiceFCNTF")
    # epochs
    epochs_range = range(EPOCHS)
    # Plotting graphs
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig("graph.png")
    plt.show()
