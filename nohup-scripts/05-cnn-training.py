#!/usr/bin/env python
# coding: utf-8

# # Convolution Neural Networks Training - JMUBEN Dataset

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import time, os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, plot_confusion_matrix
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2, InceptionV3, VGG16, EfficientNetV2S
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Add, AveragePooling2D, Dense, AvgPool2D,BatchNormalization, ReLU, DepthwiseConv2D, Reshape, Permute,Conv2D, MaxPool2D, GlobalAveragePooling2D, concatenate


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


import logging
tf.get_logger().setLevel(logging.ERROR)


# In[ ]:


tf.debugging.set_log_device_placement(False)
devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)
tf.config.experimental.set_memory_growth(devices[1], True)


# In[ ]:


ROOT_FOLDER = "/home/lais/jis-2023"
PATH_DATASETS = ROOT_FOLDER + "/datasets"
PATH_JMUBEN = PATH_DATASETS + "/jmuben"
PATH_JMUBEN_SPLITTED = PATH_JMUBEN + "/splitted"
PATH_CNN_ARTIFACTS = ROOT_FOLDER + "/cnn-artifacts"
PATH_EVALUATIONS = ROOT_FOLDER + "/evaluations"
PATH_EVALUATIONS_CNNS = PATH_EVALUATIONS + "/CNNs"


# ## 1. Defining hiperparams

# ### 1.1. Generators

# In[ ]:


BATCH_SIZE = 32
IMG_SIZE = (128, 128)
NUM_LABELS = 5
INPUT_SHAPE = (128, 128, 3)


# ### 1.2. Model

# In[ ]:


EPOCHS = 300
patience = 30


# ## 2. Defining model evaluation functions 

# In[ ]:


def get_gflops(model_h5_path):
    """Calculate GFLOPS from model."""
    
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()
        
    with graph.as_default():
        with session.as_default():
            model = tf.keras.models.load_model(model_h5_path)

            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        
            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
            flops = flops.total_float_ops
            
            gflops = flops / 1e9
            
            return gflops


# In[ ]:


def plot_graph(epochs, train_data, train_label, val_data, val_label, title, xlabel, ylabel, path, metric, i):
    """Plot line graph from model data."""
    
    plt.plot(epochs, train_data, label=train_label, color = "darkblue")
    plt.plot(epochs, val_data, label=val_label, color="cornflowerblue")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    
    if not os.path.exists(path):
        os.makedirs(path)
        
    plt.savefig(f"{path}/{i}-{metric}.jpg", dpi=115, bbox_inches="tight")
    plt.close()
    plt.clf()


# In[ ]:


def model_training_progress(history, path, i):
    """Generate visual representation from model training progress and return total training epochs."""
    
    acc = history.history["acc"]
    val_acc = history.history["val_acc"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(1, len(acc) + 1)
    
    plot_graph(epochs, acc, "Train", val_acc, "Validation", "Train and Validation Accuracy", "Epochs", "Accuracy", path, "Accuracy", i)
    plot_graph(epochs, loss, "Train", val_loss, "Validation", "Train and Validation Loss", "Epochs", "Loss", path, "Loss", i)
    
    return len(acc)


# In[ ]:


def evaluate_model(y_true, y_pred):
    """Calculate model metrics."""
    
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    
    return accuracy, f1, precision, recall


# In[ ]:


def generate_dataframe(total_params, gflops, epochs, training_time, accuracy, f1, precision, recall, i, path, model_name):
    """Generate dataframe to save model metrics."""
    
    metrics = {
        "Experiment": str(i), 
        "Total Params": total_params,
        "GFLOPS": gflops, 
        "Epochs": epochs, 
        "Training Time (sec)": training_time, 
        "Test Accuracy": accuracy, 
        "Test F1 Weightet": f1, 
        "Test Precision Weighted": precision, 
        "Test Recall Weighted": recall}
    
    df_new = pd.DataFrame(data=[metrics])
    file_path = f"{path}/metrics_{model_name}.csv"
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df = df.append(df_new)
        df.to_csv(file_path, header=True, index=False)
    else:
        df_new.to_csv(file_path, header=True, index=False)


# In[ ]:


def confusion_matrix_scorer(model, class_names, test_generator, y_true_test, path, model_name, i):
    """Generate confusion matrix from model predictions."""
    
    class estimator:
        _estimator_type = ''
        classes_= []

        def __init__(self, model, classes):
            self.model = model
            self._estimator_type = "classifier"
            self.classes_ = classes

        def predict(self, X):
            y_prob= self.model.predict(X)
            y_pred = y_prob.argmax(axis=1)
            return y_pred


    classifier = estimator(model, class_names)
    
    cm = plot_confusion_matrix(estimator=classifier, X=test_generator, y_true=y_true_test, xticks_rotation=45, cmap="Blues")
    cm.ax_.set_title(f"Confusion Matrix - {model_name}")
    cm.ax_.set_xlabel("Predicted labels")
    cm.ax_.set_xticklabels(class_names)
    cm.ax_.set_ylabel("True labels")
    cm.ax_.set_yticklabels(class_names)
    
    file_path = f"{path}/{i}-ConfusionMatrix.jpg"
    
    plt.savefig(file_path, dpi=115, bbox_inches="tight")
    plt.close()
    plt.clf()


# ## 3. Defining image generator function

# In[ ]:


def image_data_generator(train_dir, validation_dir, test_dir):
    """Construct image data generator."""
    
    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical")

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical")

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=1,
        class_mode="categorical",
        shuffle = False)
    
    return train_generator, validation_generator, test_generator


# ## 4. Training CNN models

# ### 4.1. Defining functions to return the models to train with adjustments to output layers to match the problem

# #### MobileNetV2

# In[ ]:


def mobilenetv2():
    mobilenetv2 = tf.keras.applications.MobileNetV2(
        input_shape=INPUT_SHAPE, 
        alpha=1.0,
        include_top=True,
        weights=None,
        input_tensor=None,
        pooling="max",
        classes=NUM_LABELS,
        classifier_activation=None)

    last = mobilenetv2.layers[-2]
    x = layers.Flatten()(last.output)
    x = layers.Dense(512, activation="relu")(x)
    predictions = layers.Dense(NUM_LABELS, activation="sigmoid")(x)
    
    mobilenetv2_architecture = models.Model(mobilenetv2.input, predictions)
    return mobilenetv2_architecture


# #### ShuffleNet

# - Defining architecture

# In[ ]:


def channel_shuffle(x, groups):  
    _, width, height, channels = x.get_shape().as_list()
    group_ch = channels // groups

    x = Reshape([width, height, group_ch, groups])(x)
    x = Permute([1, 2, 4, 3])(x)
    x = Reshape([width, height, channels])(x)
    
    return x


# In[ ]:


def shuffle_unit(x, groups, channels,strides):
    y = x
    x = Conv2D(channels//4, kernel_size = 1, strides = (1,1),padding = 'same', groups=groups)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = channel_shuffle(x, groups)
    
    x = DepthwiseConv2D(kernel_size = (3,3), strides = strides, padding = 'same')(x)
    x = BatchNormalization()(x)

    if strides == (2,2):
        channels = channels - y.shape[-1]
        
    x = Conv2D(channels, kernel_size = 1, strides = (1,1),padding = 'same', groups=groups)(x)
    x = BatchNormalization()(x)

    if strides ==(1,1):
        x =Add()([x,y])
        
    if strides == (2,2):
        y = AvgPool2D((3,3), strides = (2,2), padding = 'same')(y)
        x = concatenate([x,y])
    
    x = ReLU()(x)

    return x


# In[ ]:


def ShuffleNet(n_classes, start_channels, input_shape=(224, 224, 3)):
    groups = 2
    input = Input(input_shape)

    x =  Conv2D (24,kernel_size=3,strides = (2,2), padding = 'same', use_bias = True)(input)
    x =  BatchNormalization()(x)
    x =  ReLU()(x)
    
    x = MaxPool2D (pool_size=(3,3), strides = 2, padding='same')(x)

    repetitions = [3,7,3]

    for i,repetition in enumerate(repetitions):
        channels = start_channels * (2**i)

        x  = shuffle_unit(x, groups, channels,strides = (2,2))

        for i in range(repetition):
            x = shuffle_unit(x, groups, channels,strides=(1,1))

    x = GlobalAveragePooling2D()(x)

    output = Dense(n_classes,activation='softmax')(x)

    model = Model(input, output)
    
    return model


# - Model

# In[ ]:


def shufllenet():
    shufflenet = ShuffleNet(NUM_LABELS, 200, INPUT_SHAPE)

    last = shufflenet.layers[-2]
    x = layers.Flatten()(last.output)
    x = layers.Dense(512, activation="relu")(x)
    predictions = layers.Dense(NUM_LABELS, activation="sigmoid")(x)
    
    shufflenet_architecture = models.Model(shufflenet.input, predictions)
    return shufflenet_architecture


# #### InceptionV3

# In[ ]:


def inceptionv3():
    inceptionv3 = tf.keras.applications.InceptionV3(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=INPUT_SHAPE,
        pooling=max,
        classes=NUM_LABELS,
        classifier_activation=None)

    last = inceptionv3.layers[-2]
    x = layers.Flatten()(last.output)
    x = layers.Dense(512, activation="relu")(x)
    predictions = layers.Dense(NUM_LABELS, activation="sigmoid")(x)
    
    inceptionv3_architecture = models.Model(inceptionv3.input, predictions)
    return inceptionv3_architecture


# #### VGG-16

# In[ ]:


def vgg16():
    vgg16 = tf.keras.applications.VGG16(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=INPUT_SHAPE,
        pooling=max,
        classes=NUM_LABELS,
        classifier_activation=None)

    last = vgg16.layers[-2]
    x = layers.Flatten()(last.output)
    x = layers.Dense(512, activation="relu")(x)
    predictions = layers.Dense(NUM_LABELS, activation="sigmoid")(x)
    
    vgg16_architecture = models.Model(vgg16.input, predictions)
    return vgg16_architecture


# #### EfficientNetV2S

# In[ ]:


def efficientnet():
    efficientnet = tf.keras.applications.InceptionV3(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=INPUT_SHAPE,
        pooling=max,
        classes=NUM_LABELS,
        classifier_activation=None)

    last = efficientnet.layers[-2]
    x = layers.Flatten()(last.output)
    x = layers.Dense(512, activation="relu")(x)
    predictions = layers.Dense(NUM_LABELS, activation="sigmoid")(x)
    
    efficientnet_architecture = models.Model(efficientnet.input, predictions)
    return efficientnet_architecture


# #### Saving model names

# In[ ]:


model_names = ["MobileNetV2", "ShuffleNet", "InceptionV3", "VGG-16", "EfficientNetV2S"]


# ### 4.2. Making three experiments

# In[ ]:


for i in range(1, 4):
    print(f">> EXPERIMENT {i}")
    
    for model_name in model_names:
        # Experiment directory
        experiment = f"experiment{i}"
        train_dir = f"{PATH_JMUBEN_SPLITTED}/{experiment}/train"
        validation_dir = f"{PATH_JMUBEN_SPLITTED}/{experiment}/val"
        test_dir = f"{PATH_JMUBEN_SPLITTED}/{experiment}/test"

        # Generators
        train_generator, validation_generator, test_generator = image_data_generator(train_dir, validation_dir, test_dir)

        # Verifying if artifacts directory exists
        ARTIFACTS_PATH = f"{PATH_CNN_ARTIFACTS}/{model_name}"
        logs_path = f"{ARTIFACTS_PATH}/logs"
        checkpoints_path = f"{ARTIFACTS_PATH}/checkpoints"

        if not os.path.isdir(ARTIFACTS_PATH):
            os.makedirs(ARTIFACTS_PATH)

        if not os.path.isdir(logs_path):
            os.makedirs(logs_path)

        if not os.path.isdir(checkpoints_path):
            os.makedirs(checkpoints_path)

        # Callbacks
        csv_logger = callbacks.CSVLogger(f"{logs_path}/training{i}.log")
        early_stopping = callbacks.EarlyStopping(monitor="val_acc", patience=patience)
        model_checkpoint = callbacks.ModelCheckpoint(
            filepath=f"{checkpoints_path}/checkpoint{i}", 
            monitor="val_acc", 
            save_best_only=True, 
            save_weights_only=True, 
            mode="max")
        
        # Creating and compiling model
        model_architecture = mobilenetv2()
        
        if "Shuffle" in model_name:
            model_architecture = shufllenet()
        elif "VGG" in model_name:
            model_architecture = vgg16()
        elif "Inception" in model_name:
            model_architecture = inceptionv3()
        elif "Efficient" in model_name:
            model_architecture = efficientnet()
            
        model_architecture.compile(loss="categorical_crossentropy", optimizer=optimizers.RMSprop(learning_rate=1e-4), metrics=["acc"])
        model_architecture.summary()

        # Training model and verifying training time
        print(f"\n>> Training {model_name} model...")
        beginning = time.time()

        history = model_architecture.fit(
            train_generator, 
            steps_per_epoch=100, 
            epochs=EPOCHS, 
            validation_data=validation_generator, 
            validation_steps=50, 
            callbacks=[csv_logger, early_stopping, model_checkpoint],
            verbose=0)

        end = time.time()
        training_time = end - beginning
        
        # Saving the best model
        model_architecture.save(f"{ARTIFACTS_PATH}/{model_name}_exp{i}.h5")
        print("Best model saved.")
        
        # Getting GFLOPS
        gflops = get_gflops(f"{ARTIFACTS_PATH}/{model_name}_exp{i}.h5")

        # Model progress
        model_evaluation_path = f"{PATH_EVALUATIONS_CNNS}/{model_name}"
        total_epochs = model_training_progress(history, model_evaluation_path, i)
        print("Progress model saved.")
        
        # Loading model architecture and weights
        model_architecture_json = model_architecture.to_json()
        model_architecture = tf.keras.models.model_from_json(model_architecture_json)
        model_architecture.load_weights(f"{checkpoints_path}/checkpoint{i}")
        
        # Testing model
        print(f"\nTesting {model_name} model...")
        STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
        test_generator.reset()
        
        y_pred = model_architecture.predict(test_generator, steps=STEP_SIZE_TEST, verbose=1)
        y_pred = np.argmax(y_pred, axis=1)
        
        # Mapping predicted integers to labels 
        labels = (train_generator.class_indices)
        labels = dict((value, key) for key, value in labels.items())

        y_pred = [labels[element] for element in y_pred]

        y_true = test_generator.filenames
        y_true = [element[:element.find('/')] for element in y_true]
        
        total_params = model_architecture.count_params()
        
        # Evaluating model and saving metrics
        accuracy, f1, precision, recall = evaluate_model(y_true, y_pred)
        generate_dataframe(total_params, gflops, total_epochs, training_time, accuracy, f1, precision, recall, i, model_evaluation_path, model_name)
        print("Model evaluated.")

        # Plotting confusion matrix
        class_names = os.listdir(test_dir)

        true_labels = []

        for filename in y_true:
            for key, value in labels.items():
                if filename == value:
                    true_labels.append(key)

        confusion_matrix_scorer(model_architecture, class_names, test_generator, true_labels, model_evaluation_path, model_name, i)
        print("Confusion Matrix saved.")
        print("\n\n\n")

