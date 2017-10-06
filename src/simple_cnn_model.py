import pandas as pd
import numpy as np
import csv

from sklearn.cross_validation import train_test_split

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils


# DATA LOADING
data = pd.read_csv('../datasets/train.csv')
number_labels = pd.DataFrame(data['label'], columns=['label'])
number_labels = number_labels.values.ravel()  # Flatten out the classification labels
del data['label']
submission_data = pd.read_csv('../datasets/test.csv')

# DATA PRE PROCESSING
# Normalize pixel values between 0 and 1
data /= 255
submission_data /= 255
data = data.as_matrix()
data = data.reshape(data.shape[0], 1, 28, 28)
submission_data = submission_data.as_matrix()
submission_data = submission_data.reshape(submission_data.shape[0], 1, 28, 28)

# DATA SPLIT
train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(data, number_labels, stratify=number_labels)
train_data_x = np.array(train_data_x)
test_data_x = np.array(test_data_x)
train_data_y = np.array(train_data_y)
test_data_y = np.array(test_data_y)

# Reshape output labels for neural nets
train_data_y = np_utils.to_categorical(train_data_y, num_classes=10)
test_data_y = np_utils.to_categorical(test_data_y, num_classes=10)

# CREATE MODEL
model = Sequential()
model.add(Convolution2D(32, (5, 5), activation='relu', input_shape=(1, 28, 28), data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

# MODEL COMPILATION AND TRAINING
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_data_x, train_data_y, batch_size=128, epochs=10, verbose=1)

# EVALUATION
score = model.evaluate(test_data_x, test_data_y, verbose=1)
print 'Test score:' + str(score[0])
print 'Test accuracy:' + str(score[1])

# SAVE TRAINED MODEL
json_model = model.to_json()
with open("simple_cnn_model.json", "w") as json_file:
    json_file.write(json_model)
model.save_weights("simple_cnn_model_weights.h5")


# PREDICTION AND SUBMISSION
out = csv.writer(open('Submission.csv', 'w'), delimiter=',', quoting=csv.QUOTE_ALL)
out.writerow(['ImageId', 'Label'])
predictions = model.predict(submission_data)
for i in range(len(predictions)):
    row = np.array(predictions[i])
    predicted_number = row.argmax()
    out.writerow([i + 1, predicted_number])