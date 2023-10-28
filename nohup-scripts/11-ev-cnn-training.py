#!/usr/bin/env python
# coding: utf-8

# # ShuffleNet Training - Extended Validation

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import time, os, shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, plot_confusion_matrix
from tensorflow.keras import layers, models, optimizers, callbacks
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

PATH_BRACOL = PATH_DATASETS + "/bracol"
PATH_BRACOL_SPLITTED = PATH_BRACOL + "/resized_splitted"

PATH_PLANT_PATOLOGIES = PATH_DATASETS + "/plant_patologies"
PATH_PLANT_PATOLOGIES_SPLITTED = PATH_PLANT_PATOLOGIES + "/splitted"

PATH_ROCOLE = PATH_DATASETS + "/rocole"
PATH_ROCOLE_SPLITTED = PATH_ROCOLE + "/splitted"

PATH_EV_ARTIFACTS = ROOT_FOLDER + "/cnn-artifacts/extended-validation-artifacts"
PATH_EVALUATIONS = ROOT_FOLDER + "/evaluations"
PATH_EVALUATIONS_EV = PATH_EVALUATIONS + "/extended-validation"


# ## 1. Defining hiperparams

# ### 1.1. Generators

# In[ ]:


BATCH_SIZE = 32
IMG_SIZE = (128, 128)
NUM_LABELS_BRACOL = 5
NUM_LABELS_PLANT_PATOLOGIES = 2
NUM_LABELS_ROCOLE = 3
INPUT_SHAPE = (128, 128, 3)


# ### 1.2. Model

# In[ ]:


EPOCHS = 300
patience = 30


# ## 2. Defining model evaluation functions 

# In[ ]:


def plot_graph(epochs, train_data, train_label, val_data, val_label, title, xlabel, ylabel, path, metric, early_stopping):
    """Plot line graph from model data."""
    
    plt.plot(epochs, train_data, label=train_label, color="darkblue")
    plt.plot(epochs, val_data, label=val_label, color="cornflowerblue")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    if early_stopping:
        plt.savefig(f"{path}/{metric}-ES.jpg", dpi=115, bbox_inches="tight")
    else:
        plt.savefig(f"{path}/{metric}.jpg", dpi=115, bbox_inches="tight")
        
    plt.close()
    plt.clf()


# In[ ]:


def model_training_progress(history, path, early_stopping=True):
    """Generate visual representation from model training progress and return total training epochs."""
    
    acc = history.history["acc"]
    val_acc = history.history["val_acc"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(1, len(acc) + 1)
    
    plot_graph(epochs, acc, "Train", val_acc, "Validation", "Train and Validation Accuracy", "Epochs", "Accuracy", path, "Accuracy", early_stopping)
    plot_graph(epochs, loss, "Train", val_loss, "Validation", "Train and Validation Loss", "Epochs", "Loss", path, "Loss", early_stopping)
    
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


def generate_dataframe(early_stopping, epochs, accuracy, f1, precision, recall, path, dataset):
    """Generate dataframe to save model metrics."""
    
    metrics = {
        "Early Stopping": early_stopping, 
        "Epochs": epochs, 
        "Test Accuracy": accuracy, 
        "Test F1 Weightet": f1, 
        "Test Precision Weighted": precision, 
        "Test Recall Weighted": recall}
    
    df_new = pd.DataFrame(data=[metrics])
    file_path = f"{path}/metrics_{dataset}.csv"
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df = df.append(df_new)
        df.to_csv(file_path, header=True, index=False)
    else:
        df_new.to_csv(file_path, header=True, index=False)


# In[ ]:


def confusion_matrix_scorer(model, class_names, test_generator, y_true_test, path, early_stopping):
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
    cm.ax_.set_title(f"Confusion Matrix")
    cm.ax_.set_xlabel("Predicted labels")
    cm.ax_.set_xticklabels(class_names)
    cm.ax_.set_ylabel("True labels")
    cm.ax_.set_yticklabels(class_names)
    
    file_path = f"{path}/ConfusionMatrix-ES.jpg"
    
    if not early_stopping:
        file_path = f"{path}/ConfusionMatrix.jpg"
    
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


# ## 4. Training ShuffleNet

# ### 4.1. Defining functions to return model to train with adjustments to output layers to match the problem

# #### Defining architecture

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


# - BRACOL

# In[ ]:


def sn_bracol():
    shufflenet = ShuffleNet(NUM_LABELS_BRACOL, 200, INPUT_SHAPE)

    last = shufflenet.layers[-2]
    x = layers.Flatten()(last.output)
    x = layers.Dense(512, activation="relu")(x)
    predictions = layers.Dense(NUM_LABELS_BRACOL, activation="sigmoid")(x)
    
    shufflenet_architecture = models.Model(shufflenet.input, predictions)
    return shufflenet_architecture


# - Plant Patologies

# In[ ]:


def sn_plant_patologies():
    shufflenet = ShuffleNet(NUM_LABELS_PLANT_PATOLOGIES, 200, INPUT_SHAPE)

    last = shufflenet.layers[-2]
    x = layers.Flatten()(last.output)
    x = layers.Dense(512, activation="relu")(x)
    predictions = layers.Dense(NUM_LABELS_PLANT_PATOLOGIES, activation="sigmoid")(x)
    
    shufflenet_architecture = models.Model(shufflenet.input, predictions)
    return shufflenet_architecture


# - RoCoLe

# In[ ]:


def sn_rocole():
    shufflenet = ShuffleNet(NUM_LABELS_ROCOLE, 200, INPUT_SHAPE)

    last = shufflenet.layers[-2]
    x = layers.Flatten()(last.output)
    x = layers.Dense(512, activation="relu")(x)
    predictions = layers.Dense(NUM_LABELS_ROCOLE, activation="sigmoid")(x)
    
    shufflenet_architecture = models.Model(shufflenet.input, predictions)
    return shufflenet_architecture


# #### Defining model to each dataset

# In[ ]:


models_datasets = {
    "bracol": PATH_BRACOL_SPLITTED, 
    "plant_patologies": PATH_PLANT_PATOLOGIES_SPLITTED, 
    "rocole": PATH_ROCOLE_SPLITTED
}


# ### 4.2. Adjusting cnn-artifacts folder to save extended validation artifacts

# In[ ]:


jmuben_artifacts = ROOT_FOLDER + "/cnn-artifacts/jmuben"

if not os.path.exists(jmuben_artifacts):
    os.makedirs(jmuben_artifacts)

for directory in os.listdir(ROOT_FOLDER + "/cnn-artifacts"):
    if directory != "jmuben":
        source = ROOT_FOLDER + f"/cnn-artifacts/{directory}"
        destination = jmuben_artifacts + f"/{directory}"
        shutil.move(source, destination)


# ### 4.3. Training ShuffleNet in each dataset

# In[ ]:


def train_test_early_stopping(model, train_generator, step_per_epoch, validation_generator, validation_steps, callbacks, artifacts_path, evaluations_path, checkpoints_path, dataset, early_stopping):
    history = model.fit(
        train_generator, 
        steps_per_epoch=step_per_epoch, 
        epochs=EPOCHS, 
        validation_data=validation_generator, 
        validation_steps=validation_steps, 
        callbacks=callbacks,
        verbose=0)

    # Saving the best model    
    if early_stopping:
        model.save(f"{artifacts_path}/ShuffleNet_{dataset}_ES.h5")
    else:
        model.save(f"{artifacts_path}/ShuffleNet_{dataset}.h5")
    print("Best model saved.")

    # Model progress
    model_evaluation_path = f"{evaluations_path}/{dataset}"
    total_epochs = int
    
    if early_stopping:
        total_epochs = model_training_progress(history, model_evaluation_path)
    else:
        total_epochs = model_training_progress(history, model_evaluation_path, False)
        
    print("Progress model saved.")

    # Loading model architecture and weights
    model_json = model.to_json()
    model = tf.keras.models.model_from_json(model_json)
    
    if early_stopping:
        model.load_weights(f"{checkpoints_path}/checkpoint_ES")
    else:
        model.load_weights(f"{checkpoints_path}/checkpoint")

    # Testing model
    print(f"\nTesting ShuffleNet model...")
    STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
    test_generator.reset()

    y_pred = model.predict(test_generator, steps=STEP_SIZE_TEST, verbose=1)
    y_pred = np.argmax(y_pred, axis=1)

    # Mapping predicted integers to labels 
    labels = (train_generator.class_indices)
    labels = dict((value, key) for key, value in labels.items())

    y_pred = [labels[element] for element in y_pred]

    y_true = test_generator.filenames
    y_true = [element[:element.find('/')] for element in y_true]

    # Evaluating model and saving metrics
    accuracy, f1, precision, recall = evaluate_model(y_true, y_pred)
    generate_dataframe(early_stopping, total_epochs, accuracy, f1, precision, recall, model_evaluation_path, dataset)
    print("Model evaluated.")

    # Plotting confusion matrix
    class_names = os.listdir(test_dir)

    true_labels = []

    for filename in y_true:
        for key, value in labels.items():
            if filename == value:
                true_labels.append(key)

    confusion_matrix_scorer(model, class_names, test_generator, true_labels, model_evaluation_path, early_stopping)
    print("Confusion Matrix saved.")
    print("\n\n\n")


# In[ ]:


for dataset in models_datasets:
    path = models_datasets[dataset]
    
    # Directory
    train_dir = f"{path}/train"
    validation_dir = f"{path}/val"
    test_dir = f"{path}/test"
    
    # Generators
    train_generator, validation_generator, test_generator = image_data_generator(train_dir, validation_dir, test_dir)
    
    # Verifying if artifacts directory exists
    ARTIFACTS_PATH = f"{PATH_EV_ARTIFACTS}/{dataset}"
    checkpoints_path = f"{ARTIFACTS_PATH}/checkpoints"

    if not os.path.isdir(ARTIFACTS_PATH):
        os.makedirs(ARTIFACTS_PATH)
    
    if not os.path.isdir(checkpoints_path):
        os.makedirs(checkpoints_path)
    
    # Callbacks - Early Stopping
    csv_logger_es = callbacks.CSVLogger(f"{ARTIFACTS_PATH}/training_ES.log")
    early_stopping = callbacks.EarlyStopping(monitor="val_acc", patience=patience)
    model_checkpoint_es = callbacks.ModelCheckpoint(
        filepath=f"{checkpoints_path}/checkpoint_ES", 
        monitor="val_acc", 
        save_best_only=True, 
        save_weights_only=True, 
        mode="max")
    
    # Callbacks - No Early Stopping
    csv_logger = callbacks.CSVLogger(f"{ARTIFACTS_PATH}/training.log")
    model_checkpoint = callbacks.ModelCheckpoint(
        filepath=f"{checkpoints_path}/checkpoint", 
        monitor="val_acc", 
        save_best_only=True, 
        save_weights_only=True, 
        mode="max")
    
    # Defining parameters based on dataset
    step_per_epoch = len(train_generator)
    validation_steps = len(validation_generator)
    
    # Creating and compiling model
    shuffle_net = sn_bracol()
        
    if "plant" in dataset:
        shuffle_net = sn_plant_patologies()
    elif "rocole" in dataset:
        shuffle_net = sn_rocole()
        
    shuffle_net.compile(loss="categorical_crossentropy", optimizer=optimizers.RMSprop(learning_rate=1e-4), metrics=["acc"])
    shuffle_net.summary()
    
    # Train with Early Stopping callback
    callbacks_es = [csv_logger_es, early_stopping, model_checkpoint_es]
    train_test_early_stopping(shuffle_net, train_generator, step_per_epoch, validation_generator, validation_steps, callbacks_es, ARTIFACTS_PATH, PATH_EVALUATIONS_EV, checkpoints_path, dataset, True)
    
    # Creating and compiling model
    shuffle_net = sn_bracol()
        
    if "plant" in dataset:
        shuffle_net = sn_plant_patologies()
    elif "rocole" in dataset:
        shuffle_net = sn_rocole()
        
    shuffle_net.compile(loss="categorical_crossentropy", optimizer=optimizers.RMSprop(learning_rate=1e-4), metrics=["acc"])
    shuffle_net.summary()
    
    # Train without Early Stopping callback
    callbacks_no_es = [csv_logger, model_checkpoint]
    train_test_early_stopping(shuffle_net, train_generator, step_per_epoch, validation_generator, validation_steps, callbacks_no_es, ARTIFACTS_PATH, PATH_EVALUATIONS_EV, checkpoints_path, dataset, False)

