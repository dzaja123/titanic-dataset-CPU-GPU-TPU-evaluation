# Import Libraries
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import urllib.request
import time
import psutil

# Ignore the pandas "depricated "warnings
import warnings
warnings.filterwarnings("ignore")


# Load Data Function
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
    except:
        print("File not found")
        data = None
    return data

# Preprocess Data Function
def preprocess_data(data, is_test=False):
    # Preprocessing
    gender_factorized = data.copy()

    # Factorize 'Sex'
    gender_factorized['Sex'] = gender_factorized['Sex'].replace(['male', 'female'], [0, 1])

    # Factorize 'Embarked'
    gender_factorized['Embarked'] = gender_factorized['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2])

    if not is_test:
        # Select features for training data
        gender_factorized['Survived'] = gender_factorized['Survived'].astype(int)
        feature = gender_factorized[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
        feature = feature.dropna()
        return feature
    else:
        # Select features for test data
        feature = gender_factorized[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
        feature['Fare'].replace(np.nan, feature['Fare'].median(), inplace=True)
        feature['Age'].replace(np.nan, feature['Age'].median(), inplace=True)
        # Add a placeholder for 'Survived' column (not used in predictions)
        feature.loc[:, 'Survived'] = 0
        return feature

# Split Data Function
def split_data(feature_data):
    # Splitting Data
    def split_data_frame(df, frac, axis=0) -> list:
        if axis == 0:
            threshold = int(df.shape[0] * frac)
            part1 = df.iloc[0: threshold, :].reset_index(drop=True)
            part2 = df.iloc[threshold:, :].reset_index(drop=True)
        elif axis == 1:
            threshold = df.shape[1] * frac
            part1 = df.iloc[:, 0: threshold].reset_index(drop=True)
            part2 = df.iloc[:, threshold:].reset_index(drop=True)
        else:
            print("Key 'axis' is '0' or '1'")
            return [None, None]
        return [part1, part2]

    splitted = split_data_frame(feature_data, 0.8)
    train_data = np.array(splitted[0].iloc[:, 1:])
    train_label = np.array(splitted[0].iloc[:, 0])
    validation_data = np.array(splitted[1].iloc[:, 1:])
    validation_label = np.array(splitted[1].iloc[:, 0])
    return train_data, train_label, validation_data, validation_label

# Build Model Function
def build_model():
    # Building Model
    model = Sequential()
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Train and Evaluate on CPU Function
def train_evaluate_cpu(train_data, train_label, validation_data, validation_label, epochs=500):
    print("Training and evaluating on CPU:")
    model_cpu = build_model()
    acc, loss = train_model(model_cpu, train_data, train_label, epochs)
    print("Training Accuracy:", acc[-1])
    print("Training Loss:", loss[-1])

    result = evaluate_model(model_cpu, validation_data, validation_label)
    print("Validation Accuracy:", result[1])
    print("Validation Loss:", result[0])
    print("------------------------- \n")
    return model_cpu

# Train and Evaluate on GPU Function
def train_evaluate_gpu(train_data, train_label, validation_data, validation_label, epochs=500):
    print("Training and evaluating on GPU:")
    with tf.device('/device:GPU:0'):
        model_gpu = build_model()
        acc_gpu, loss_gpu = train_model(model_gpu, train_data, train_label, epochs)
        print("Training Accuracy:", acc_gpu[-1])
        print("Training Loss:", loss_gpu[-1])

        result_gpu = evaluate_model(model_gpu, validation_data, validation_label)
        print("Validation Accuracy:", result_gpu[1])
        print("Validation Loss:", result_gpu[0])
        print("------------------------- \n")
        return model_gpu

# Train and Evaluate on TPU Function
def train_evaluate_tpu(train_data, train_label, validation_data, validation_label, epochs=500):
    print("Training and evaluating on TPU:")
    tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
    tf.config.experimental_connect_to_cluster(tpu_resolver)
    tf.tpu.experimental.initialize_tpu_system(tpu_resolver)
    strategy = tf.distribute.TPUStrategy(tpu_resolver)

    with strategy.scope():
        model_tpu = build_model()
        acc_tpu, loss_tpu = train_model(model_tpu, train_data, train_label, epochs)
        print("Training Accuracy:", acc_tpu[-1])
        print("Training Loss:", loss_tpu[-1])

        result_tpu = evaluate_model(model_tpu, validation_data, validation_label)
        print("Validation Accuracy:", result_tpu[1])
        print("Validation Loss:", result_tpu[0])
        print("------------------------- \n")
        return model_tpu

# Train Model Function
def train_model(model, train_data, train_label, epochs=500):
    history = model.fit(train_data, train_label, epochs=epochs, verbose=0)
    acc = history.history['accuracy']
    loss = history.history['loss']
    return acc, loss

# Evaluate Model Function
def evaluate_model(model, validation_data, validation_label):
    result = model.evaluate(validation_data, validation_label)
    return result

# Predict Data Function
def predict_data(model, test_data):
    predict = model.predict(test_data)
    binary_result = (predict > 0.5).astype(int)
    binary_result = binary_result.reshape(test_data.shape[0])
    return binary_result

def download_dataset(train_data_path, test_data_path):    
    # Check if files are downloaded, if not, download them
    if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
        print("Downloading files...")

        # URLs of the dataset files
        train_url = "https://raw.githubusercontent.com/dzaja123/titanic-dataset-CPU-GPU-TPU-evaluation/main/dataset/train.csv"
        test_url = "https://raw.githubusercontent.com/dzaja123/titanic-dataset-CPU-GPU-TPU-evaluation/main/dataset/test.csv"

        # Download the files
        urllib.request.urlretrieve(train_url, train_data_path)
        urllib.request.urlretrieve(test_url, test_data_path)

# Main Function
def main():
    # Load Data
    train_data_path = "dataset/train.csv"
    test_data_path = "dataset/test.csv"

    # Check if dataset files exist, if not, download them
    download_dataset(train_data_path, test_data_path)

    train_data = load_data(train_data_path)
    test_data = load_data(test_data_path)

    # Preprocess Train Data
    feature_data = preprocess_data(train_data)

    # Split Data
    train_data, train_label, validation_data, validation_label = split_data(feature_data)

    # Check if running in Colab
    try:
        in_colab = "google.colab" in str(get_ipython())
    except:
        print("Runing the code locally.")
        in_colab = False

    # Train and Evaluate based on the available device
    if in_colab and (tf.test.gpu_device_name() == "/device:GPU:0"):
        # Train and Evaluate on GPU
        start_time = time.time()
        model = train_evaluate_gpu(train_data, train_label, validation_data, validation_label)
        training_time = time.time() - start_time
        print("GPU Training Time:", training_time)

        # Evaluate Inference Time on GPU
        start_time = time.time()
        _ = predict_data(model, validation_data)
        inference_time_gpu = time.time() - start_time
        print("GPU Inference Time:", inference_time_gpu)

        # Evaluate Memory Usage on GPU
        gpu_memory_usage = psutil.virtual_memory().used
        print("GPU Memory Usage:", gpu_memory_usage / (1024 * 1024), "MB")

    elif in_colab and ("COLAB_TPU_ADDR" in os.environ):
        # Train and Evaluate on TPU
        start_time = time.time()
        model = train_evaluate_tpu(train_data, train_label, validation_data, validation_label)
        training_time = time.time() - start_time
        print("TPU Training Time:", training_time)

        # Evaluate Inference Time on TPU
        start_time = time.time()
        _ = predict_data(model, validation_data)
        inference_time_tpu = time.time() - start_time
        print("TPU Inference Time:", inference_time_tpu)

        # Evaluate Memory Usage on TPU
        tpu_memory_usage = psutil.virtual_memory().used
        print("TPU Memory Usage:", tpu_memory_usage / (1024 * 1024), "MB")

    else:
        # Train and Evaluate on CPU
        start_time = time.time()
        model = train_evaluate_cpu(train_data, train_label, validation_data, validation_label)
        training_time = time.time() - start_time
        print("CPU Training Time:", training_time)

        # Evaluate Inference Time on CPU
        start_time = time.time()
        _ = predict_data(model, validation_data)
        inference_time_cpu = time.time() - start_time
        print("CPU Inference Time:", inference_time_cpu)

        # Evaluate Memory Usage on CPU
        cpu_memory_usage = psutil.virtual_memory().used
        print("CPU Memory Usage:", cpu_memory_usage / (1024 * 1024), "MB")

    # Preprocess Test Data
    test_data_processed = preprocess_data(test_data, is_test=True)
    test_data_array = np.array(test_data_processed.iloc[:, 1:])

    # Prediction
    binary_result = predict_data(model, test_data_array)

    # Print predictions
    print("Predictions: \n", binary_result)

    # Clear the session to release resources
    keras.backend.clear_session()

if __name__ == "__main__":
    main()
