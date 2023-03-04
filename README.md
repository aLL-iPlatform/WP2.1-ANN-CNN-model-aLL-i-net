# WP2.1-ANN-CNN-model-aLL-i-net
Deep learning model using tensor flow/azure, set up at cloud
Use TensorFlow to build a model on digital twin replicated against physical surrogate  model using current data sets in an open source framework.
TensorFlow is a powerful open source library for building machine learning models and allows for flexibility in designing models of various complexities. It provides numerous APIs to create models from pre-trained models, and to use datasets from different sources. With TensorFlow, you can build a neural network model capable of replicating the physical models from the current datasets and use it to create a digital twin.


 Python script to build a new dataset called aLL-i using current datasets, developed and trained at the internet:

# Import the necessary libraries
import tensorflow as tf
import numpy as np

# Load the current datasets from the internet
train_data = np.load('train_data.npy')
test_data = np.load('test_data.npy')

# Create a neural network model using TensorFlow
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(train_data.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Fit the model to the training dataset
model.fit(train_data, epochs=10,


# Evaluate the performance of the model on the test dataset
test_loss, test_acc = model.evaluate(test_data)

print('Test accuracy:', test_acc)

# Create a new dataset called aLL-i using the model
new_data = model.predict(test_data)
np.save('aLL-i.npy', new_data)

print('New dataset aLL-i created successfully!')




