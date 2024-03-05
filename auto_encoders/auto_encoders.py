from numpy import shape
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model

# Assuming train_detagen and test_detagen are defined elsewhere
train_detagen=ImageDataGenerator(rescale=1./255)
test_detagen=ImageDataGenerator(rescale=1./255)

train_gen = train_detagen.flow_from_directory(
    'E:/KLU/3rd year/3_2/deep learning/Deep Learning Programs/train',
    target_size=(28,28),
    batch_size=32,
    class_mode='input'
)

test_gen = test_detagen.flow_from_directory(
    'E:/KLU/3rd year/3_2/deep learning/Deep Learning Programs/test',
    target_size=(28,28),
    batch_size=32,
    class_mode='input'
)

imp_img = Input(shape=(28,28,3))
x = Conv2D(32, (3,3), activation='relu', padding='same')(imp_img)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(16, (3,3), activation='relu', padding='same')(x)
enc = MaxPooling2D((2,2), padding='same')(x)

x = Conv2D(16, (3,3), activation='relu', padding='same')(enc)
x = UpSampling2D((2,2))(x)
x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
dec = Conv2D(3, (3,3), activation='sigmoid', padding='same')(x)

autoencoder = Model(imp_img, dec)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = autoencoder.fit(train_gen, epochs=10, validation_data=test_gen)

loss = autoencoder.evaluate_generator(test_gen)
print("Test loss:", loss)