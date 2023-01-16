{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "FJUcpGTTsDoy"
   },
   "source": [
    "# Real and Fake product image detection"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "wJyV4VGHsDo5"
   },
   "source": [
    "### Import Keras and tensorflow"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "OVauBbnZsDo3"
   },
   "outputs": [],
   "source": [
    "#pip install keras\n",
    "#pip install tensorflow\n",
    "\n",
    "from keras.models import Sequential\n",
    "from keras.layers import Conv2D, Flatten, Dense,MaxPool2D\n",
    "import numpy as np\n",
    "from keras.preprocessing import image\n",
    "from keras.callbacks import EarlyStopping, ModelCheckpoint\n",
    "import tensorflow as tf\n",
    "from keras.preprocessing.image import ImageDataGenerator\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline\n",
    "try:\n",
    "    from tensorflow.python.util import module_wrapper as deprecation\n",
    "except ImportError:\n",
    "    from tensorflow.python.util import deprecation_wrapper as deprecation\n",
    "deprecation._PER_MODULE_WARNING_LIMIT = 0"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "wJyV4VGHsDo5"
   },
   "source": [
    "### Dataset import with image generator function"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "Ar8sS26UsDo6",
    "outputId": "67df878e-a795-479c-b3a4-7d8b59d25111"
   },
   "outputs": [],
   "source": [
    "# import dataset using the generator function\n",
    "n_batch = 128\n",
    "trainset_datagenerator = ImageDataGenerator(\n",
    "    rescale=1./255,\n",
    "    rotation_range=10.,\n",
    "    width_shift_range=0.1,\n",
    "    height_shift_range=0.1,\n",
    "    zoom_range=0.2,\n",
    "    horizontal_flip=True)\n",
    "\n",
    "testset_datagenerator = ImageDataGenerator(rescale = 1./255)\n",
    "\n",
    "training_image_set = trainset_datagenerator.flow_from_directory('product/training',\n",
    "                                                 target_size=(128,128),\n",
    "                                                 batch_size =n_batch,\n",
    "                                                 class_mode = 'binary')\n",
    "\n",
    "testing_image_set = testset_datagenerator.flow_from_directory('product/test',\n",
    "                                            target_size=(128,128),\n",
    "                                            batch_size =n_batch,\n",
    "                                            class_mode = 'binary')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_labels = training_image_set.labels\n",
    "print(train_labels)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "x = training_image_set\n",
    "print(x)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Plot between instance and class"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 312
    },
    "id": "GzMnO0irsDo6",
    "outputId": "3db013b7-66a8-42f5-8212-c9838e45937b",
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# Plot the chart between the number of instances and the no of class\n",
    "p1 = plt.hist(training_image_set.classes, bins=range(0,3), alpha=0.8, color='green', edgecolor='black')\n",
    "p2 = plt.hist(testing_image_set.classes,  bins=range(0,3), alpha=0.8, color='black', edgecolor='black')\n",
    "plt.ylabel('No of Instances')\n",
    "plt.xlabel('No of Class')\n",
    "plt.title('View Plot between instance and class')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Verify the data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 840
    },
    "id": "wKRsNH5BsDo7",
    "outputId": "9c646e17-f0f0-41b1-bc2a-024d71655e71"
   },
   "outputs": [],
   "source": [
    "# To verify that the dataset looks correct, let's plot the few images from the training set and display the class name below each image:\n",
    "for P, q in training_image_set:\n",
    "    print(P.shape, q.shape)\n",
    "    plt.figure(figsize=(18,18))\n",
    "    for K in range(16):\n",
    "        plt.subplot(5,5,K+1)\n",
    "        plt.axis('off')\n",
    "        plt.title('Label: Product Image ')\n",
    "        product_image = np.uint8(255*P[K,:,:,0])\n",
    "        plt.imshow(product_image, cmap='gray')\n",
    "    break"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "b5T550zysDo8"
   },
   "source": [
    "## CNN Architectutres "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def history_model_fit(model):\n",
    "    if model == model1:\n",
    "        history = model.fit(training_image_set, epochs=10, \n",
    "                    validation_data=(training_image_set))\n",
    "    else:\n",
    "        history = model.fit(training_image_set, epochs=4, \n",
    "                    validation_data=(training_image_set))\n",
    "\n",
    "    \n",
    "    return history\n",
    "\n",
    "def loss_chart(history):\n",
    "    plt.figure(0)\n",
    "    plt.plot(history.history['loss'], label='Loss')\n",
    "    plt.plot(history.history['val_loss'], label='Validation Loss')\n",
    "    plt.legend()\n",
    "    plt.title('Training - Loss Function')\n",
    "    \n",
    "def accuracy_chart(history):\n",
    "    plt.figure(1)\n",
    "    plt.plot(history.history['accuracy'], label='Accuracy')\n",
    "    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')\n",
    "    plt.legend(loc = 'upper right')\n",
    "    plt.title('Train - Accuracy')\n",
    "    \n",
    "def accuracy_output(model):\n",
    "      \n",
    "    test_loss, test_acc = model.evaluate(testing_image_set, verbose=2)\n",
    "    return test_acc\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "code below define the convolutional base using a common pattern"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 1)  LeNet-5 Complete Architecture"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "145VxxufsDo9"
   },
   "outputs": [],
   "source": [
    "# The lines of code below define the convolutional base using a common pattern: a stack of Conv2D and MaxPooling2D layers.\n",
    "model1 = Sequential()\n",
    "model1.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128,128,3)))\n",
    "model1.add(MaxPool2D(pool_size=(2,2)))\n",
    "model1.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))\n",
    "model1.add(MaxPool2D(pool_size=(2,2)))\n",
    "model1.add(Conv2D(128, kernel_size=(3, 3),activation='relu'))\n",
    "model1.add(MaxPool2D(pool_size=(2,2)))\n",
    "\n",
    "# Add Dense layers on top\n",
    "model1.add(Flatten())\n",
    "model1.add(Dense(activation=\"relu\", units=256))\n",
    "model1.add(Dense(activation=\"sigmoid\", units=1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "oEDKNE-yIzLj",
    "outputId": "5dfe3117-646b-40ea-fc59-d1bf9397c84c",
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# Checking model summary and the complete architecture of your model:\n",
    "\n",
    "model1.summary()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "_KE5xUKasDo-"
   },
   "source": [
    "### Compile  and train the model"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Metrics\n",
    "This for log, we choose accuracy ."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "Z0Rz31RdsDo-"
   },
   "outputs": [],
   "source": [
    "# here, the created sequential model has been compiled to calculate the loss as well as accuracy\n",
    "model1.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "aWOmblZ-sDo_"
   },
   "outputs": [],
   "source": [
    "# Train the model process has been start here to evaluate that the created model can be used further in real time within the application.\n",
    "train_model = [\n",
    "    EarlyStopping(monitor='val_loss', patience=10),\n",
    "    ModelCheckpoint(filepath='/content/drive/MyDrive/product/model.h5', monitor='val_loss', save_best_only=True, mode ='max'),\n",
    "]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "training_labels = training_image_set.class_indices\n",
    "#testing_labels = testing_image_set.class_indices\n",
    "print(training_labels)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "_t4DDztRsDo_",
    "outputId": "03630286-9a84-4616-a6d8-2344a94d7f63"
   },
   "outputs": [],
   "source": [
    "# hist = model.fit(\n",
    "#         training_image_set,\n",
    "#         steps_per_epoch=80,\n",
    "#         epochs=80,\n",
    "#         validation_data=testing_image_set,\n",
    "#         validation_steps=28,\n",
    "#         callbacks = train_model\n",
    "#     )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "history1 = history_model_fit(model1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Evaluate the model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot the figure for loss and the validation loss\n",
    "loss_chart(history1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# plot the figure for accuracy and validation accuracy\n",
    "accuracy_chart(history1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Display the accuracy obtained\n",
    "test_acc = accuracy_output(model1)\n",
    "print(test_acc)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install visualkeras\n",
    "import visualkeras\n",
    "visualkeras.layered_view(model1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2) VGGNET Architecture\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# VGGNET Architecture\n",
    "\n",
    "from keras import layers\n",
    "from keras.models import Model, Sequential\n",
    "\n",
    "from functools import partial\n",
    "\n",
    "conv3 = partial(layers.Conv2D,\n",
    "                kernel_size=3,\n",
    "                strides=1,\n",
    "                padding='same',\n",
    "                activation='relu')\n",
    "\n",
    "def block(in_tensor, filters, n_convs):\n",
    "    conv_block = in_tensor\n",
    "    for _ in range(n_convs):\n",
    "        conv_block = conv3(filters=filters)(conv_block)\n",
    "    return conv_block\n",
    "\n",
    "def _vgg(in_shape=(227,227,3),\n",
    "         n_classes=10,\n",
    "         opt='sgd',\n",
    "         n_stages_per_blocks=[2, 2, 3, 3, 3]):\n",
    "    in_layer = layers.Input(in_shape)\n",
    "\n",
    "    block1 = block(in_layer, 64, n_stages_per_blocks[0])\n",
    "    pool1 = layers.MaxPool2D()(block1)\n",
    "    block2 = block(pool1, 128, n_stages_per_blocks[1])\n",
    "    pool2 = layers.MaxPool2D()(block2)\n",
    "    block3 = block(pool2, 256, n_stages_per_blocks[2])\n",
    "    pool3 = layers.MaxPool2D()(block3)\n",
    "    block4 = block(pool3, 512, n_stages_per_blocks[3])\n",
    "    pool4 = layers.MaxPool2D()(block4)\n",
    "    block5 = block(pool4, 512, n_stages_per_blocks[4])\n",
    "    pool5 = layers.MaxPool2D()(block5)\n",
    "    flattened = layers.GlobalAvgPool2D()(pool5)\n",
    "\n",
    "    dense1 = layers.Dense(4096, activation='relu')(flattened)\n",
    "    dense2 = layers.Dense(4096, activation='relu')(dense1)\n",
    "    preds = layers.Dense(1000, activation='sigmoid')(dense2)\n",
    "\n",
    "    model2 = Model(in_layer, preds)\n",
    "    model2.compile(loss=\"sparse_categorical_crossentropy\", optimizer='adam', metrics=[\"accuracy\"])\n",
    "    return model2\n",
    "\n",
    "def vgg16(in_shape=(227,227,3), n_classes=10, opt='sgd'):\n",
    "    return _vgg(in_shape, n_classes, opt)\n",
    "\n",
    "def vgg19(in_shape=(227,227,3), n_classes=10, opt='sgd'):\n",
    "    return _vgg(in_shape, n_classes, opt, [2, 2, 4, 4, 4])\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    model2 = vgg19()\n",
    "    print(model2.summary())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Train model\n",
    "train_model = [\n",
    "    EarlyStopping(monitor='val_loss', patience=10),\n",
    "    ModelCheckpoint(filepath='/content/drive/MyDrive/product/model2.h5', monitor='val_loss', save_best_only=True, mode ='max'),\n",
    "]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "history2 = history_model_fit(model2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Evaluate the Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot the figure for loss and the validation loss\n",
    "loss_chart(history2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# plot the figure for accuracy and validation accuracy\n",
    "accuracy_chart(history2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Display the accuracy obtained\n",
    "test_acc = accuracy_output(model2)\n",
    "print(test_acc)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "import visualkeras\n",
    "visualkeras.layered_view(model2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### AlexNet Architecture"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# AlexNet CNN Architecture\n",
    "from keras import layers\n",
    "from keras.models import Model\n",
    "\n",
    "def alexnet(in_shape=(128,128,3), n_classes=1000, opt='sgd'):\n",
    "    in_layer = layers.Input(in_shape)\n",
    "    conv1 = layers.Conv2D(96, 11, strides=4, activation='relu')(in_layer)\n",
    "    pool1 = layers.MaxPool2D(3, 2)(conv1)\n",
    "    conv2 = layers.Conv2D(256, 5, strides=1, padding='same', activation='relu')(pool1)\n",
    "    pool2 = layers.MaxPool2D(3, 2)(conv2)\n",
    "    conv3 = layers.Conv2D(384, 3, strides=1, padding='same', activation='relu')(pool2)\n",
    "    conv4 = layers.Conv2D(256, 3, strides=1, padding='same', activation='relu')(conv3)\n",
    "    pool3 = layers.MaxPool2D(3, 2)(conv4)\n",
    "    flattened = layers.Flatten()(pool3)\n",
    "    dense1 = layers.Dense(4096, activation='relu')(flattened)\n",
    "    drop1 = layers.Dropout(0.5)(dense1)\n",
    "    dense2 = layers.Dense(4096, activation='relu')(drop1)\n",
    "    drop2 = layers.Dropout(0.5)(dense2)\n",
    "    preds = layers.Dense(1000, activation='sigmoid')(drop2)\n",
    "    \n",
    "\n",
    "    model3 = Model(in_layer, preds)\n",
    "    model3.compile(loss=\"sparse_categorical_crossentropy\", optimizer='adam', metrics=[\"accuracy\"])\n",
    "\n",
    "    return model3\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    model3 = alexnet()\n",
    "    print(model3.summary())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Train model\n",
    "train_model = [\n",
    "    EarlyStopping(monitor='val_loss', patience=10),\n",
    "    ModelCheckpoint(filepath='/content/drive/MyDrive/product/model3.h5', monitor='val_loss', save_best_only=True, mode ='max'),\n",
    "]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "training_labels = training_image_set.class_indices\n",
    "#testing_labels = testing_image_set.class_indices\n",
    "print(training_labels)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "history3 = history_model_fit(model3)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##Evaluate the Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot the figure for loss and the validation loss\n",
    "loss_chart(history3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# plot the figure for accuracy and validation accuracy\n",
    "accuracy_chart(history3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Display the accuracy obtained\n",
    "test_acc = accuracy_output(model3)\n",
    "print(test_acc)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import visualkeras\n",
    "visualkeras.layered_view(model3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Comparison of the CNN architectures\n",
    "ac1 = accuracy_output(model)\n",
    "ac2 = accuracy_output(model1)\n",
    "ac3 = accuracy_output(model2)\n",
    "\n",
    "print(\"LeNet architecture accuracy:\", ac1)\n",
    "print(\"VggNet architecture accuracy:\", ac2)\n",
    "print(\"AlexNet architecture accuracy:\", ac3)\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "tiw4FPytsDpA"
   },
   "source": [
    "## Prediction process completed and ready to detect fake or real image"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.utils import image_utils"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "blPktyymsDpB"
   },
   "outputs": [],
   "source": [
    "def product_image_Prediction(loc):\n",
    "    testing_product_image = image_utils.load_img(loc, target_size = (128,128))\n",
    "    plt.axis('off')\n",
    "    plt.imshow(testing_product_image)\n",
    "    testing_product_image = image_utils.img_to_array(testing_product_image)\n",
    "    testing_product_image = np.expand_dims(testing_product_image, axis =0)\n",
    "    outcome = model.predict(testing_product_image)\n",
    "    if outcome[0][0] == 1:\n",
    "        predictions = 'This Product is Real'\n",
    "    else:\n",
    "        predictions = 'This Product is Fake'\n",
    "    print('Prediction Result: ',predictions)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 283
    },
    "id": "_yJUgJwDsDpB",
    "outputId": "808e86ea-9fff-429e-f68f-3510b77558f6"
   },
   "outputs": [],
   "source": [
    "product_image_path = input(\"Please enter product image path to check weather it is real or fake: \")\n",
    "test_image_1 = product_image_Prediction(product_image_path)\n",
    "# product/test/fake/2732.jpg"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 283
    },
    "id": "oWYqAvfZsDpB",
    "outputId": "97737b10-b265-40f4-e8cd-20d48a1225e1"
   },
   "outputs": [],
   "source": [
    "product_image_path = input(\"Please enter product image path to check weather it is real or fake: \")\n",
    "test_image_2 = product_image_Prediction(product_image_path)\n",
    "# product/test/fake/2742.jpg"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 283
    },
    "id": "H04V1a1jsDpC",
    "outputId": "bc20068c-3f6b-4847-eb94-39c8644ce01b"
   },
   "outputs": [],
   "source": [
    "product_image_path = input(\"Please enter product image path to check weather it is real or fake: \")\n",
    "test_image_3 = product_image_Prediction(product_image_path)\n",
    "# product/test/fake/5459.jpg"
   ]
  }
 ],
 "metadata": {
  "colab": {
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
