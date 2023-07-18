
import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

# declaring input(samples) and output(labels) lists
train_labels = []
train_samples = []

# Question:

# An experimental drug was tested on individuals from age 13 to 100
# Trail has 2100 participants. Half under 65 and Half 65 and above.
# Around 95% of patients 65 or older had side effects
# Around 95 % of patients under 65 had no side effects

# taking randomised input

for i in range(50):
    # The ~5% of young people which had side effects
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(1)

    # The ~5% of old people which had no side effects
    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(0)

for i in range(1000):
    # The ~95% of young people which had no side effects
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(0)

    # The ~95% of old people which had side effects
    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(1)

train_samples = np.array(train_samples)
train_labels = np.array(train_labels)
# shuffle the samples & labels
# MOST IMPORTANT STEP!!!!!!!!!!!!!!!
train_labels, train_samples = shuffle(train_labels, train_samples)

# this causes the samples to be in range of 0 to
# 1 for easy processing by neural network
# also reshape into 2D array as it is the required shape by the function
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1, 1))

# for i in scaled_train_samples:
#     print(i)

model = Sequential([
    Dense(units=16 , input_shape=(1,) , activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')
])

# details about model
# model.summary()

# compiling model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# training the model
# model.fit(x=scaled_train_samples,y=train_labels,batch_size=10,epochs=30
#           ,shuffle=True,verbose=2)


# creation of validation set for the model to give accurate results on general
# examples
# here WE ADDED validation_set=0.1 parameter 
# model.fit(x=scaled_train_samples,y=train_labels,validation_split=0.1,
#           batch_size=10,epochs=30
#           ,shuffle=True,verbose=2)


# Test data has the same procedure as the training data
test_samples = []
test_labels = []
# taking randomised input

for i in range(50):
    # The ~5% of young people which had side effects
    random_younger = randint(13, 64)
    test_labels.append(1)
    test_samples.append(random_younger)

    # The ~5% of old people which had no side effects
    random_older = randint(65, 100)
    test_samples.append(random_older)
    test_labels.append(0)

for i in range(1000):
    # The ~95% of young people which had no side effects
    random_younger = randint(13, 64)
    test_samples.append(random_younger)
    test_labels.append(0)

    # The ~95% of old people which had side effects
    random_older = randint(65, 100)
    test_samples.append(random_older)
    test_labels.append(1)

test_samples = np.array(test_samples)
test_labels = np.array(test_labels)
test_labels, test_samples = shuffle(test_labels, test_samples)
scaled_test_samples = scaler.fit_transform(test_samples.reshape(-1, 1))


# predictions on the model

predictions = model.predict(x=scaled_test_samples,batch_size=10,verbose=0)
# for i in predictions:
#     print(i)
# index=0 => no side effect
# index=1 => side effect

rounded_predictions=np.argmax(predictions,axis=-1)
# for i in rounded_predictions:
#     print(i)


