import matplotlib.pyplot as plt

def display_images(test_images, predicted_images, num_images=5):
    fig, axs = plt.subplots(2, num_images, figsize=(10, 10))
    for i in range(num_images):
        # Actual
        axs[0, i].imshow(test_images[i])
        axs[0, i].set_title("ORIGINAL")
        axs[0, i].axis("off")
        # Predicted
        axs[1, i].imshow(predicted_images[i])
        axs[1, i].set_title("PREDICTED")
        axs[1, i].axis("off")
    plt.tight_layout()
    plt.show(block=True)