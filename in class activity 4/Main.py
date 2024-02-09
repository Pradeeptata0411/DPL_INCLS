import preprocessing as mp
import Model as mm
if __name__ == "__main__":
    images_folder_path = 'E:\\KLU\\3rd year\\3_2\deep learning\\Deep Learning Programs\\train'
    imdata = mp.PreProcess_Data()
    imdata.visualization_images(images_folder_path, 2)
    imagefile, label, df = imdata.preprocess(images_folder_path)
    csv_file_path = 'output.csv'
    df.to_csv(csv_file_path, index=False)
    print(f"CSV file saved at: {csv_file_path}")
    # train, test = imdata.generate_train_test_images(imagefile, label)
    # csv_file_path2 = 'output_test.csv'
    # test.to_csv(csv_file_path2, index=False)
    # print(f"CSV file saved at: {csv_file_path2}")
    tr_gen, tt_gen,va_gen = imdata.generate_train_test_images(imagefile, label)
    print("train Generator :-", tr_gen)
    print("test Generator :-", tt_gen)
    print("validation Generator :-", va_gen)

    print()
    print()
    print()
    print()
    print()
    print()

    CnnModel = mm.DeepANN()
    model1 = CnnModel.CNN_MODEL()
    print("train generator ", tr_gen)
    ANN_history = model1.fit(tr_gen, epochs=10, validation_data=va_gen)

    Ann_test_loss, Ann_test_acc = model1.evaluate(tr_gen)
    #Ann_test_loss, Ann_test_acc = model1.evaluate(tt_gen)
    print(f'Test accuracy: {Ann_test_acc}')
    model1.save("my_model1.keras")
    print("The ANN architecture is ")
    print(model1.summary())
