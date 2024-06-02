import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Load the data
data_dict = pickle.load(open('data.pickle', 'rb'))
data = data_dict['data']
labels = np.array(data_dict['labels'])

# Check the keys in the dictionary and their contents
print(data_dict.keys())
print(data_dict)

# Check the length of each element in data
lengths = [len(item) for item in data]
print(f"Min length: {min(lengths)}, Max length: {
      max(lengths)}, Average length: {np.mean(lengths)}")

# Pad sequences to make them equal in length
max_length = max(lengths)
data_padded = pad_sequences(data, maxlen=max_length,
                            padding='post', dtype='float32')
print(f"Padded data shape: {data_padded.shape}")

# One-hot encoding
unique_labels = list(set(labels))
label_to_index = {label: index for index, label in enumerate(unique_labels)}
labels_indexed = np.array([label_to_index[label] for label in labels])
labels_categorical = to_categorical(labels_indexed)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(
    data_padded, labels_categorical, test_size=0.2, shuffle=True, random_state=42)

# Build the model
model = Sequential()
model.add(Dense(64, input_shape=(x_train.shape[1],), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(unique_labels), activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model directly with fit
history = model.fit(x_train, y_train, epochs=10,
                    batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model's performance
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title('Model Accuracy')
plt.show()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Model Loss')
plt.show()
