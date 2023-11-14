import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set the paths to your dataset
dataset_path = 'path/to/your/dataset'
authorized_path = os.path.join(dataset_path, 'authorized')
unauthorized_path = os.path.join(dataset_path, 'unauthorized')

# Get the list of image files
authorized_images = [os.path.join(authorized_path, img) for img in os.listdir(authorized_path) if img.endswith('.jpg')]
unauthorized_images = [os.path.join(unauthorized_path, img) for img in os.listdir(unauthorized_path) if img.endswith('.jpg')]

# Create labels (1 for authorized, 0 for unauthorized)
authorized_labels = np.ones(len(authorized_images))
unauthorized_labels = np.zeros(len(unauthorized_images))

# Combine authorized and unauthorized data
all_images = authorized_images + unauthorized_images
all_labels = np.concatenate([authorized_labels, unauthorized_labels])

# Shuffle the data
indices = np.arange(len(all_images))
np.random.shuffle(indices)

all_images = np.array(all_images)[indices]
all_labels = all_labels[indices]

# Create a data generator for loading and augmenting images
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Create data generators for training and validation
train_generator = datagen.flow_from_dataframe(
    pd.DataFrame({'image_path': all_images, 'label': all_labels}),
    x_col='image_path',
    y_col='label',
    subset='training',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

validation_generator = datagen.flow_from_dataframe(
    pd.DataFrame({'image_path': all_images, 'label': all_labels}),
    x_col='image_path',
    y_col='label',
    subset='validation',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Create a new model for fine-tuning
new_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = tf.keras.Sequential([
    new_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=5, validation_data=validation_generator)
#TODO : svae the model