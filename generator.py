import os
import numpy as np
import cv2
import tensorflow as tf

from keras.src.preprocessing.image import ImageDataGenerator
from keras.src.utils import pad_sequences
from sklearn import preprocessing


class Generator(tf.keras.utils.Sequence):
    def __init__(self, DATASET_PATH, BATCH_SIZE=32, shuffle_images=True, subset="training",
                 validation_split=0.2):
        """ Initialize Generator object.

        Args:
            DATASET_PATH: Path to folder containing individual folders named by their class names
            BATCH_SIZE: The size of the batches to generate.
            shuffle_images: If True, shuffles the images read from the DATASET_PATH
            image_min_side: After resizing the minimum side of an image is equal to image_min_side.
        """
        self.dataset_Path = DATASET_PATH
        self.encoded_image_labels = []
        self.label_lengths = []
        self.max_label_len = 0
        self.original_texts = []
        self.subset = subset
        self.validation_split = validation_split
        self.batch_size = BATCH_SIZE
        self.final_batch_size = BATCH_SIZE * 2
        self.shuffle_images = shuffle_images
        self.load_image_paths_labels(DATASET_PATH)
        self.create_image_groups()
        self.datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,  # Normalize pixel values to [0, 1]
            rotation_range=40,    # Random rotation between -40 and 40 degrees
            width_shift_range=0.2,  # Random horizontal shift
            height_shift_range=0.2, # Random vertical shift
            shear_range=0.2,       # Shear angle in counter-clockwise direction
            zoom_range=0.2,        # Random zoom between 80% and 120%
            horizontal_flip=True,  # Random horizontal flip
            brightness_range=[0.5, 1.5]  # Random brightness adjustment
        )

    def load_image_paths_labels(self, dataset_path):
            """
            Load image paths and labels from the handwriting dataset.
            """
            self.image_paths = []
            self.image_labels = []

            with open('/home/mdelab/PycharmProjects/FCNforCV/archive/iam_words/words.txt') as f:
                contents = f.readlines()

            lines = [line.strip() for line in contents][18:]  # Adjust the index if needed

            for line in lines:
                try:
                    splits = line.split(' ')
                    status = splits[1]

                    if status == 'ok':
                        word_id = splits[0]
                        word = "".join(splits[8:])
                        encoded_label = self.encode_to_labels(word)
                        splits_id = word_id.split('-')
                        filepath = os.path.join(dataset_path, 'words', splits_id[0],
                                                '{}-{}'.format(splits_id[0], splits_id[1]),
                                                word_id + '.png')

                        if os.path.exists(filepath):
                            self.image_paths.append(filepath)
                            self.image_labels.append(word)
                            self.encoded_image_labels.append(encoded_label)
                except:
                    pass

            # Split the data into training and validation sets
            split_index = int(len(self.image_paths) * (1 - self.validation_split))
            if self.subset == "training":
                self.image_paths = self.image_paths[:split_index]
                self.image_labels = self.image_labels[:split_index]
            elif self.subset == "validation":
                self.image_paths = self.image_paths[split_index:]
                self.image_labels = self.image_labels[split_index:]

            assert len(self.image_paths) == len(self.image_labels)



    def resize_image(self, img):
        # Resize image to 32x128 while maintaining aspect ratio.
        # If the aspect ratio does not match, pad the resized image
        # to maintain the aspect ratio and fit the desired size.

        target_height = 32
        target_width = 128

        # Calculate the aspect ratio of the target dimensions
        target_aspect = target_width / target_height

        # Calculate the aspect ratio of the current image
        current_height, current_width = img.shape[:2]
        current_aspect = current_width / current_height

        if current_aspect > target_aspect:
            # If the current aspect ratio is greater than the target,
            # resize the image to fit the target width.
            new_width = target_width
            new_height = round(new_width / current_aspect)
            resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        else:
            # If the current aspect ratio is less than or equal to the target,
            # resize the image to fit the target height.
            new_height = target_height
            new_width = round(new_height * current_aspect)
            resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Pad the resized image to fit the target dimensions
        pad_height = target_height - new_height
        pad_width = target_width - new_width

        top_pad = pad_height // 2
        bottom_pad = pad_height - top_pad
        left_pad = pad_width // 2
        right_pad = pad_width - left_pad

        # Create a new image with the target dimensions and paste the resized image
        final_img = np.zeros((target_height, target_width), dtype=img.dtype)
        final_img[top_pad:top_pad+new_height, left_pad:left_pad+new_width] = resized_img

        # Add the channel dimension
        final_img = np.expand_dims(final_img, axis=-1)

        return final_img


    def load_images(self, image_group):
        images = []
        for image_path in image_group:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read the image in grayscale
            img = self.resize_image(img)

            transformed_img = self.datagen.random_transform(img)

            # Check if an extra dimension is added and remove it if necessary
            if transformed_img.ndim > img.ndim:
                transformed_img = np.squeeze(transformed_img, axis=-1)

            images.append(img)
            images.append(transformed_img)

        return images

    def create_image_groups(self):
        if self.shuffle_images:
            # Randomly shuffle dataset
            seed = 4321
            np.random.seed(seed)
            np.random.shuffle(self.image_paths)
            np.random.seed(seed)
            np.random.shuffle(self.image_labels)

        # Divide image_paths and image_labels into groups of BATCH_SIZE
        self.image_groups = [
            [
                self.image_paths[x % len(self.image_paths)]
                for x in range(i, i + self.batch_size)
            ]
            for i in range(0, len(self.image_paths), self.batch_size)
        ]

        self.label_groups = []
        self.encoded_label_groups = []

        for i in range(0, len(self.image_labels), self.batch_size):
            batch = []
            ebatch = []
            for x in range(i, i + self.batch_size):
                label = self.image_labels[x % len(self.image_labels)]
                enclabel = self.encoded_image_labels[x % len(self.encoded_image_labels)]
                for k in range(2):
                    batch.append(label)
                    ebatch.append(enclabel)
            self.label_groups.append(batch)
            self.encoded_label_groups.append(ebatch)


    def encode_to_labels(self, txt):
        char_list = "!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        # Encoding each output word into digits
        dig_lst = []
        for index, chara in enumerate(txt):
            dig_lst.append(char_list.index(chara))

        return dig_lst

    def construct_image_batch(self, image_group):
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        # construct an image batch object
        image_batch = np.zeros((self.final_batch_size,) + max_shape, dtype='float32')

        # copy all images to the upper left part of the image batch object
        for image_index, image in enumerate(image_group):
            image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

        return image_batch


    def __len__(self):
        """
        Number of batches for generator.
        """

        return len(self.image_groups)

    def __getitem__(self, index):
        # Generate one batch of data
        image_group = self.image_groups[index]
        label_group = self.label_groups[index]
        encoded_label_group = self.encoded_label_groups[index]
        images = self.load_images(image_group)
        image_batch = self.construct_image_batch(images)

        # Pad label sequences
        padded_labels = pad_sequences(encoded_label_group, maxlen=self.max_label_len, padding='post', value=-1)

        return np.array(image_batch), np.array(padded_labels)

if __name__ == "__main__":

    BASE_PATH = 'dataset'
    train_dir = "/home/mdelab/PycharmProjects/FCNforCV/archive/iam_words/"
    BATCH_SIZE = 8
    train_generator = Generator(train_dir, BATCH_SIZE, shuffle_images=True, subset="training")
    val_generator = Generator(train_dir, BATCH_SIZE, shuffle_images=True, subset="validation")
    print(len(train_generator))
    print(len(val_generator))
    image_batch, label_group = train_generator.__getitem__(0)
    print(image_batch.shape)
    print(label_group.shape)
