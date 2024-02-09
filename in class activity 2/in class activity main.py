import inclassactivity2
from inclassactivity2 import PreProcess_Data

if __name__ == "__main__":
    images_folder_path = 'E:\\KLU\\3rd year\\3_2\deep learning\\Deep Learning Programs\\train'
    imdata = PreProcess_Data()
    imdata.visualization_images(images_folder_path, 4)
    train, label, df = imdata.preprocess(images_folder_path)
    csv_file_path = 'output.csv'
    df.to_csv(csv_file_path, index=False)
    print(f"CSV file saved at: {csv_file_path}")