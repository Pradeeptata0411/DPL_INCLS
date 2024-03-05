import tensorflow.python.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from torch.nn.modules import padding

class Autoencoder(tensorflow.keras.Model):
    def annmodel(self):
        train_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
        'E:/KLU/3rd year/3_2/deep learning/Deep Learning Programs/train',
        target_size=(28,28),
        batch_size=32,
        class_mode='input'
        )

        test_generator = test_datagen.flow_from_directory(
        'E:/KLU/3rd year/3_2/deep learning/Deep Learning Programs/test',
        target_size=(28,28),
        batch_size=32,
        class_mode='input'
        )

        input_img = Input(shape=(28,28,3))
        x = Conv2D(32,(3,3),activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2,2), padding='same')(x)
        x = Conv2D(16,(3,3),activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2,2), padding='same')(x)

        x = Conv2D(16, (3,3),activation='relu',padding='same')(encoded)
        x = UpSampling2D((2,2))(x)
        x = Conv2D(32,(3,3),activation='relu', padding='same')(x)
        x = UpSampling2D((2,2))(x)
        decoded = Conv2D(3,(3,3),activation='sigmoid',padding='same')(x)


        autoencoder = Model(input_img,decoded)
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return autoencoder,train_generator,test_generator
