import numpy as np
import os
import matplotlib.image as im
import tensorflow as tf


if __name__ == "__main__":
    # Get data
    data_arr = list[np.ndarray]()

    for image in os.listdir("DATA"):
        img_data = im.imread(f"DATA/{image}").ravel()

        # Turn data into 100x100 unraveled array
        data_arr.append(np.append(img_data, np.ones(10000 - len(img_data))))

    data = np.asmatrix(data_arr)

    # Normilize data with train data parameters
    train_mean = np.load("andria_kilasonia_data_mean.npy")
    data -= train_mean

    # Reduce data dimensions to fit the model params
    u_reduce = np.load("andria_kilasonia_reduce_matrix.npy")
    data = data * u_reduce

    # Get neural network model
    model = tf.keras.models.load_model("andria_kilasonia_nn_model")

    # Get predictions
    predictions = model.predict(data).argmax(axis=1) + 1

    # Save predictions
    output_file = open("andia_kilasonia.txt", "w")

    for i, image in enumerate(os.listdir("DATA")):
        image = image[:-4]
        output_file.write(f"{image}-{predictions[i]}\n")

    output_file.close()
