import os

from roboflow import Roboflow

from oral.labels import Labels


class RoboflowData(Labels):
    def __init__(self, base_data_path=None):
        super().__init__(base_data_path)
        self.rf = Roboflow(api_key=os.getenv('RF_API_KEY'))

    def fetch(self, rf):
        project = rf.workspace("oraldisease").project("oral-diseases")
        version = project.version(2)
        dataset = version.download("yolov5", location=self.base_data_path, overwrite=True)
        self.load_classes()


if __name__ == '__main__':
    rf_data = RoboflowData()
    rf_data.fetch(rf_data.rf, '../data')
    # rf_data.tag_images(tags={}, base_data_path='../data/oral-diseases-2', category='train')
    # print(rf_data.dataset.location)
