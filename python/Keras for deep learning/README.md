## Keras for DL
![image](https://github.com/DevJSter/AIML/assets/115056248/799e224f-9018-42e8-ae48-bb50e9b41b55)
# Why Keras?

Keras was developed with a focus on enabling fast experimentation. Because of this, it's very user-friendly and allows us to go from idea to implementation with only a few steps.

Aside from this, users often wonder why choose Keras over another neural network API?

Our general advice is to learn more than one neural network API. Once you have the core deep learning concepts down from the Deep Learning Fundamentals course, you can implement your knowledge in code across several APIs.

Yes, the syntactical details for implementation will differ slightly between APIs, but once you learn one, it is relatively easy to catch on to another. Sometimes, one API may have an advantage over the others for a particular implementation.

Especially for job prospects, knowing more than one API will put more tools in your skill set, and demonstrating this will make you a more valuable candidate.

## TensorFlow Integration

Keras was originally created by François Chollet. Historically, Keras was a high-level API that sat on top of one of three lower-level neural network APIs and acted as a wrapper for these lower-level libraries. These libraries were referred to as Keras backend engines.

You could choose TensorFlow, Theano, or CNTK as the backend engine you'd like to work with.

- TensorFlow
- Theano
- CNTK

Ultimately, TensorFlow became the most popular backend engine for Keras.

Later, Keras became integrated with the TensorFlow library and now comes completely packaged with it.
![image](https://github.com/DevJSter/AIML/assets/115056248/1dffd894-cb16-4d4d-9ed2-61f80d6d5b0a)
# Differences In Imports

From a usability standpoint, many changes between the older way of using Keras with a configured backend versus the new way of having Keras integrated with TensorFlow is in the import statements.

For example, previously, we could access the Dense module from Keras with the following import statement.

```python
from keras.layers import Dense
```

Now, using Keras with TensorFlow, the import statement looks like this:

```python
from tensorflow.keras.layers import Dense
```

Below, you can see the difference between the old way and the new way of importing some common Keras modules.

## Before TensorFlow Integration

```python
import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
```

## After TensorFlow Integration

```python
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, BatchNormalization, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```

# How To Install Keras

Since Keras now comes packaged with TensorFlow, we need to install TensorFlow with the command:

```bash
pip install tensorflow
```

That's it!
# GPU Support for TensorFlow

TensorFlow code, including Keras, seamlessly runs on a single GPU without explicit code configuration. GPU support is available for Ubuntu and Windows systems with CUDA-enabled cards.

## Hardware Requirements

The only requirement is an NVIDIA GPU card with CUDA Compute Capability. Check the TensorFlow website for supported versions.

### Linux Setup

For Linux, TensorFlow recommends using a Docker image with GPU support, simplifying installation with NVIDIA GPU drivers.

### Windows Setup

1. **Install TensorFlow:** Use `pip install tensorflow` and ensure Microsoft Visual C++ redistributable is installed.

2. **Install Nvidia Drivers:** Download and install Nvidia drivers from the Nvidia website.

3. **Install CUDA Toolkit:** Choose a version compatible with TensorFlow and install it. Ensure Visual Studio is installed if prompted.

4. **Install CuDNN SDK:** Create an account on Nvidia's website, download cuDNN corresponding to the CUDA Toolkit version, and follow installation steps.

5. **Verify GPU Detection:** In a Jupyter notebook or IDE, run the code:

   ```python
   import tensorflow as tf
   print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
   ```

   If the output is 1, TensorFlow has identified your GPU successfully.

   Note: If the output is 0, check for errors in the console, and verify the CUDA environment variable as discussed in the cuDNN installation steps. Restart your machine and try again.

Once TensorFlow detects your GPU, your future code will run on the GPU by default.
# Data Processing for Neural Network Training

## Samples and Labels

To train a neural network in a supervised learning task, we need a dataset of samples and corresponding labels. Samples are individual data points, while labels are associated classifications.

For example, in sentiment analysis, headlines could have labels like "positive" or "negative." For images of cats and dogs, labels might be "cat" or "dog."

In deep learning, samples are often called input data, and labels are referred to as target data.

## Expected Data Format

Before training a neural network, we must ensure data is in a format compatible with the model. The Sequential model from Keras, integrated with TensorFlow, expects input data (`x`) in one of the following formats:

- Numpy array or list of arrays
- TensorFlow tensor or list of tensors
- Dict mapping input names to corresponding arrays/tensors
- tf.data dataset returning (inputs, targets) or (inputs, targets, sample_weights)
- Generator or keras.utils.Sequence returning (inputs, targets) or (inputs, targets, sample_weights)

Labels (`y`) should match the format of input data.

## Process Data in Code
```
import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

# Create empty lists for input and target data
train_labels = []
train_samples = []

# Data Creation
for i in range(50):
    # ~5% of younger individuals who experienced side effects
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(1)

    # ~5% of older individuals who did not experience side effects
    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(0)

for i in range(1000):
    # ~95% of younger individuals who did not experience side effects
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(0)

    # ~95% of older individuals who experienced side effects
    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(1)

# Convert lists to numpy arrays and shuffle
train_labels = np.array(train_labels)
train_samples = np.array(train_samples)
train_labels, train_samples = shuffle(train_labels, train_samples)

# Data Processing
# Scale data to range from 0 to 1 using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1, 1))
```

```python
# Create an Artificial Neural Network with TensorFlow's Keras API

## Code Setup

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
```

Check GPU availability:

```python
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)
```

## Build a Sequential Model

```python
# Create a Sequential model
model = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')
])
```

### First Hidden Layer

- Dense layer with 16 neurons, input shape of (1,), and relu activation.

### Second Hidden Layer

- Dense layer with 32 neurons and relu activation.

### Output Layer

- Dense layer with 2 neurons (for binary classification) and softmax activation.

```python
# Display model summary
model.summary()
```

Output:

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 16)                32        
_________________________________________________________________
dense_1 (Dense)              (None, 32)                544       
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 66        
=================================================================
Total params: 642
Trainable params: 642
Non-trainable params: 0
```

The model is created using the Sequential API with three Dense layers. In the next episode, we will train this model on the previously generated data.

![image](https://github.com/DevJSter/AIML/assets/115056248/980fc866-7cee-4d62-ad40-2edfd60f61a2)


