import subprocess
import zipfile
import yaml
import os

from oral.labels import Labels


def download_dataset():
    command = "kaggle datasets download -d salmansajid05/oral-diseases"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()
    print("Command executed, return code:", process.returncode)


def extract_dataset(data_dir: str):
    with zipfile.ZipFile("oral-diseases.zip", "r") as zip_ref:
        zip_ref.extractall(data_dir)


class KaggleData(Labels):
    def __init__(self):
        base_data_path = os.path.join('data',
                                      'Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset',
                                      'Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset',
                                      'Data')
        super().__init__(base_data_path)

        print("Current working directory:", os.getcwd())
        print("Absolute file path:", os.path.abspath(os.path.join(self.base_data_path, 'data.yaml')))

        with open(os.path.join(self.base_data_path, 'data.yaml'), 'r') as file:
            data = yaml.safe_load(file)
            self.classes = data['names']

        self.id2label = {}
        for i, label in enumerate(data['names']):
            self.id2label[i] = label

    def fetch(self):
        if not os.path.exists("oral-diseases.zip"):
            print('Downloading dataset')
            download_dataset()
            print('Extracting dataset')
            extract_dataset(os.getenv("DATA_DIR"))
        else:
            print('Dataset already downloaded, extracting...')
            extract_dataset(os.getenv("DATA_DIR"))
        self.load_classes()

        def remove_labels_txt(txt_path: str):
            if os.path.exists(os.path.split(txt_path)[0]):
                if 'labels.txt' not in os.listdir(os.path.split(txt_path)[0]):
                    print('labels.txt already removed')
                else:
                    os.remove(txt_path)
            else:
                print('data directory does not exist')
                exit(-1)

        for category in ['train', 'val']:
            remove_labels_txt(os.path.join(self.base_data_path,
                                           'labels', category, 'labels.txt'))

        print("Dataset downloaded and extracted")


if __name__ == '__main__':
    labels = Labels()
