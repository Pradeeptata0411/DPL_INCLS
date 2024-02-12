from Preprocessing_Optimizers import PreProcess_Data
import Classification_Optimizer as cm

if __name__ == "__main__":
    images_folder_path = 'E:\\KLU\\3rd year\\3_2\\deep learning\\Deep Learning Programs\\imagefile'
    imdata = PreProcess_Data()
    imdata.visualization_images(images_folder_path, 4)
    imagefile, label, df = imdata.preprocess(images_folder_path)
    tr_gen, tt_gen, va_gen = imdata.generate_train_test_images(imagefile, label)
    image_shape=(28,28,3)
    model_adam=cm.DeepANN.simple_model(image_shape,optimizer='adam')
    model_sgd=cm.DeepANN.simple_model1(image_shape,optimizer='sgd')
    model_rmsprop=cm.DeepANN.simple_model2(image_shape,optimizer='rmsprop')
    cm.compare_model([model_adam,model_sgd,model_rmsprop] ,tr_gen,va_gen,epochs=3)