from flask import Flask, render_template, request
from flask.helpers import url_for
from Preprocessing_Optimizers import PreProcess_Data
import Classification_Optimizer as cm
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare', methods=['POST'])
def compare():
    if request.method == 'POST':
        optimizer1 = request.form['optimizer1']
        optimizer2 = request.form['optimizer2']
        optimizer3 = request.form['optimizer3']

        images_folder_path = 'E:\\KLU\\3rd year\\3_2\\deep learning\\Deep Learning Programs\\imagefile'
        imdata = PreProcess_Data()
        imdata.visualization_images(images_folder_path, 2)
        imagefile, label, df = imdata.preprocess(images_folder_path)
        tr_gen, tt_gen, va_gen = imdata.generate_train_test_images(imagefile, label)
        image_shape = (28, 28, 3)

        model1 = cm.DeepANN().simple_model(optimizer=optimizer1)
        model2 = cm.DeepANN().simple_model1(optimizer=optimizer2)
        model3 = cm.DeepANN().simple_model2(optimizer=optimizer3)

        models = [model1, model2, model3]
        history_list, test_results = cm.compare_model(models, tr_gen, va_gen, epochs=3)

        # Plotting code
        plot_paths = []
        for i, history in enumerate(history_list):
            # Your plotting code here
            plt_path = BytesIO()
            plt.savefig(plt_path, format='png')
            plt_path.seek(0)
            plot_paths.append(base64.b64encode(plt_path.getvalue()).decode('utf-8'))
            plt.clf()

        image_shape = (28, 28, 3)

        model1 = cm.DeepANN.simple_model(image_shape, optimizer=optimizer1)
        model2 = cm.DeepANN.simple_model1(image_shape, optimizer=optimizer2)
        model3 = cm.DeepANN.simple_model2(image_shape, optimizer=optimizer3)

        models = [model1, model2, model3]
        history_list, test_results = cm.compare_model(models, tr_gen, va_gen, epochs=3)



if __name__ == "__main__":
    app.run(debug=True)
