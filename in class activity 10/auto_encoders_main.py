import image_disply as pre
import Classification_model as cm

def autoencoder_call():
    autoen=cm.Autoencoder()
    autoencoder_model,train_gen,test_gen=autoen.annmodel()
    autoencoder_model.fit(train_gen, epochs=5,validation_data=test_gen)
    autoencoder_model.save('autoencoder_saved.keras')
    loss = autoencoder_model.evaluate(test_gen)
    print("Test loss:",loss)
    test_images , _ = next(test_gen)
    predicted_images  = autoencoder_model.predict(test_images)
    pre.display_images(test_images,predicted_images,10)

if __name__ == "__main__":
    autoencoder_call()
