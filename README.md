Let’s create a simple neural network model using Keras to predict handwritten digits between 1 and 9. We’ll use the MNIST dataset, which contains grayscale images of single digits (0-9). Our model will consist of two fully connected layers. Here’s how you can build it:

Load the MNIST Dataset: First, load the MNIST dataset, which consists of 28x28 pixel grayscale images of handwritten digits. Each image is associated with a label representing the digit (0-9).
Define the Neural Network Model: We’ll create a sequential model using Keras. The architecture will have:
An input layer that flattens the 28x28 image into a 1D array.
A hidden layer with 128 units and ReLU activation.
An output layer with 10 units (one for each digit) and softmax activation.
Compile and Train the Model: Compile the model with an appropriate optimizer (e.g., Adam) and loss function (e.g., categorical cross-entropy). Train the model using the training data.
Evaluate the Model: Evaluate the model’s performance on a validation set. You can also explore improvements by adjusting hyperparameters or adding more layers.
Make Predictions: Once trained, use the model to predict the digit class for new images.
Here’s a simplified example using TensorFlow and Keras:

Python

import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess the data
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.4f}")

# Make predictions
predictions = model.predict(test_images[:5])
print("Predictions for the first 5 test images:")
for i, pred in enumerate(predictions):
    print(f"Image {i+1}: Predicted class {tf.argmax(pred)}")
