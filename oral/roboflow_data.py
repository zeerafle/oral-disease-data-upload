import os

from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateEntry, Region
from globox import AnnotationSet
from roboflow import Roboflow

from oral.labels import Labels


class RoboflowData(Labels):
    def __init__(self, base_data_path=None):
        super().__init__(base_data_path)
        self.rf = Roboflow(api_key=os.getenv('RF_API_KEY'))

    def load_classes(self):
        with open(os.path.join(self.base_data_path, 'train', '_darknet.labels'), 'r') as file:
            self.classes = file.read().split('\n')

        for i, label in enumerate(self.classes):
            self.id2label[i] = label

    def fetch(self, rf, overwrite=False):
        project = rf.workspace("oraldisease").project("oral-diseases")
        version = project.version(3)
        dataset = version.download("darknet", location=self.base_data_path, overwrite=overwrite)
        self.load_classes()

    def tag_images(self, tags: dict, category: str = 'train'):
        yolo_ann = AnnotationSet.from_yolo_v5(
            folder=os.path.join(self.base_data_path, category),
            image_folder=os.path.join(self.base_data_path, category),
        )

        for annotation in yolo_ann:
            # if annotation only contains 'Caries', skip
            annotation_labels = [box.label for box in annotation.boxes]
            caries = str(self.classes.index('Caries'))
            if set(annotation_labels) == {caries}:
                continue
            with open(os.path.join(self.base_data_path, category, annotation.image_id),
                      mode='rb') as image_contents:
                self.tagged_images_with_regions.append(
                    ImageFileCreateEntry(
                        name=annotation.image_id,
                        contents=image_contents.read(),
                        regions=[
                            Region(tag_id=tags[self.id2label[int(region.label)]],
                                   left=region.xmin / annotation.image_width,
                                   top=region.ymin / annotation.image_height,
                                   width=region.width / annotation.image_width,
                                   height=region.height / annotation.image_height)
                            for region in annotation.boxes
                        ]
                    )
                )


if __name__ == '__main__':
    rf_data = RoboflowData()
    rf_data.fetch(rf_data.rf)
    # rf_data.tag_images(tags={}, base_data_path='../data/oral-diseases-2', category='train')
    # print(rf_data.dataset.location)
