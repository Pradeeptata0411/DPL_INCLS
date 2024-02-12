from Preprocessing_Optimizers import PreProcess_Data
import Model as cm
import matplotlib.pyplot as plt

if __name__ == "__main__":
    images_folder_path = 'E:\\KLU\\3rd year\\3_2\\deep learning\\Deep Learning Programs\\train'
    imdata = PreProcess_Data()
    imdata.visualization_images(images_folder_path, 2)
    imagefile, label, df = imdata.preprocess(images_folder_path)
    tr_gen, tt_gen, va_gen = imdata.generate_train_test_images(imagefile, label, batch_size=100)  # Set batch_size here
    image_shape = (128, 128, 3)
    model_instance = cm.DeepANN()
    model = model_instance.Rnn_model(image_shape)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Check the number of labels to ensure they match the number of images
    print(f"Number of training images: {len(tr_gen.classes)}")
    print(f"Number of test images: {len(tt_gen.classes)}")
    print(f"Number of validation images: {len(va_gen.classes)}")

    history = model.fit(tr_gen, validation_data=va_gen, epochs=8)

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    # Plotting the training and validation accuracy
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

    loss, accuracy = model.evaluate(tt_gen)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)
