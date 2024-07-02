import yaml
import os

from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateEntry, Region
from globox import AnnotationSet
import supervision as sv


class Labels:
    def __init__(self, base_data_path: str = None, classes=None):
        self.base_data_path = base_data_path
        self.tagged_images_with_regions = []
        self.classes = classes
        self.id2label = {}

        print("Current working directory:", os.getcwd())
        print("Absolute file path:", os.path.abspath(os.path.join(self.base_data_path, 'data.yaml')))

    def load_classes(self):
        if not self.classes:
            with open(os.path.join(self.base_data_path, 'data.yaml'), 'r') as file:
                data = yaml.safe_load(file)
                self.classes = data['names']

        for i, label in enumerate(data['names']):
            self.id2label[i] = label

    def tag_images(self, tags: dict, category_first: bool = False, category: str = 'train', images_folder: str = 'images', labels_folder: str = 'labels'):
        labels_folder_list = [self.base_data_path, labels_folder]
        image_folder_list = [self.base_data_path, images_folder]
        if category_first:
            labels_folder_list.insert(1, category)
            image_folder_list.insert(1, category)
        else:
            labels_folder_list.append(category)
            image_folder_list.append(category)

        folder = os.path.join(*labels_folder_list)
        image_folder = os.path.join(*image_folder_list)

        folder = str(folder)
        image_folder = str(image_folder)

        yolo_ann = AnnotationSet.from_yolo_v5(
            folder=folder,
            image_folder=image_folder,
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
