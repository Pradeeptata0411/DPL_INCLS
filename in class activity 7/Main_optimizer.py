from Preprocessing_Optimizers import PreProcess_Data
import Model as cm

if __name__ == "__main__":
    images_folder_path = 'E:\\KLU\\3rd year\\3_2\\deep learning\\Deep Learning Programs\\train'
    imdata = PreProcess_Data()
    imdata.visualization_images(images_folder_path, 2)
    imagefile, label, df = imdata.preprocess(images_folder_path)
    tr_gen, tt_gen, va_gen = imdata.generate_train_test_images(imagefile, label)
    image_shape=(128,128,3)
    model_instance = cm.DeepANN()  # Instantiate the class
    model = model_instance.Rnn_model(image_shape)  # Call the method on the instantiated object
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(tt_gen, validation_data=va_gen, epochs=2)
    loss, accuracy = model.evaluate(tt_gen)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)
