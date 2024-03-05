import cv2

import preprocessing as mp
import Model as mm
import numpy as np
import matplotlib.pyplot as plt
import Model as cm

if __name__ == "__main__":
    images_folder_path = 'E:\\KLU\\3rd year\\3_2\deep learning\\Deep Learning Programs\\train'
    imdata = mp.PreProcess_Data()
    imdata.visualization_images(images_folder_path, 2)
    train, label, df = imdata.preprocess(images_folder_path)
    tr_gen, tt_gen,va_gen = imdata.generate_train_test_images(train, label)

    CnnModel = mm.DeepANN()
    input_shape1=(28,28,3)
    model3 = CnnModel.cnn_reg_model(input_shape=input_shape1)
    cnn = model3.fit(tr_gen, epochs=5, validation_data=va_gen)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    # plt.plot(CNN_history_mini_batch_norm.history['val_loss'], label='Validation Loss')
    plt.plot(cnn.history['loss'], label='Training Loss')
    plt.plot(cnn.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Loss')
    plt.ylabel('Accuracy')
    plt.title('cnn Loss and  Accuracy')
    plt.legend()

    # plt.subplot(1, 2, 2)
    # plt.plot(CNN_history_mini_batch_norm.history['loss'], label='Training Loss')
    # plt.plot(CNN_history_mini_batch_norm.history['val_loss'], label='Validation Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title('CNN_MODEL_miniBatchNormalization Training and Validation Loss')
    # plt.legend()
    plt.tight_layout()
    plt.show()