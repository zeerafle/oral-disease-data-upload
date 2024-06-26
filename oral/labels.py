import yaml
import os

from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateEntry, Region, Tag
from globox import AnnotationSet


class Labels:
    def __init__(self, base_data_path: str = None):
        if base_data_path is None:
            self.base_data_path = os.path.join('data',
                                               'Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset',
                                               'Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset',
                                               'Data')
        self.tagged_images_with_regions = []

        print("Current working directory:", os.getcwd())
        print("Absolute file path:", os.path.abspath(os.path.join(self.base_data_path, 'data.yaml')))

        with open(os.path.join(self.base_data_path, 'data.yaml'), 'r') as file:
            data = yaml.safe_load(file)
            self.classes = data['names']

        self.id2label = {}
        for i, label in enumerate(data['names']):
            self.id2label[i] = label

    def tag_images(self, tags: dict, category: str = 'train'):
        yolo_ann = AnnotationSet.from_yolo_v5(
            folder=os.path.join(self.base_data_path, 'labels', category),
            image_folder=os.path.join(self.base_data_path, 'images', category),
        )

        for annotation in yolo_ann:
            with open(os.path.join(self.base_data_path, 'images', category, annotation.image_id),
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
    labels = Labels()
    # labels.tag_images()
