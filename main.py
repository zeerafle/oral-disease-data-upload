import subprocess
import os
import zipfile
from dotenv import load_dotenv

from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import CustomVisionErrorException, ImageFileCreateBatch
from msrest.authentication import ApiKeyCredentials

from oral.kaggle_data import KaggleData
from oral.labels import Labels
from oral.roboflow_data import RoboflowData

load_dotenv()

ENDPOINT = os.getenv("VISION_TRAINING_ENDPOINT")
TRAINING_KEY = os.getenv("VISION_TRAINING_KEY")
DOMAIN_ID = os.getenv("DOMAIN_ID")
PROJECT_ID = os.getenv("PROJECT_ID")


def upload_images(labels: Labels):
    for i in range(0, len(labels.tagged_images_with_regions), 64):
        print(i)
        batch = labels.tagged_images_with_regions[i:i + 64]
        upload_result = trainer.create_images_from_files(PROJECT_ID, ImageFileCreateBatch(images=batch))
        if not upload_result.is_batch_successful:
            print("Image batch upload failed")
            for image in upload_result.images:
                print("Image status: ", image.status)


def get_tags(labels: Labels, cv_trainer: CustomVisionTrainingClient):
    tags = []
    for class_name in labels.classes:
        try:
            tags.append(cv_trainer.create_tag(PROJECT_ID, class_name))
        except CustomVisionErrorException:
            print(f"Tag {class_name} already exists")
            continue

    if not tags:
        print("Tags already exist")
        tags = cv_trainer.get_tags(PROJECT_ID)

    return {tag.name: tag.id for tag in tags}


if __name__ == "__main__":
    credentials = ApiKeyCredentials(in_headers={"Training-key": TRAINING_KEY})
    trainer = CustomVisionTrainingClient(ENDPOINT, credentials)

    kaggle_data = KaggleData()
    kaggle_data.fetch()
    tag_name_to_id = get_tags(kaggle_data, trainer)
    kaggle_data.tag_images(tags=tag_name_to_id)
    kaggle_data.tag_images(tags=tag_name_to_id, category='val')
    upload_images(kaggle_data)

    roboflow_data = RoboflowData(os.path.join(os.getenv("DATA_DIR"), 'oral-diseases-2'))
    roboflow_data.fetch(roboflow_data.rf)
    roboflow_data.tag_images(tags=tag_name_to_id)
    roboflow_data.tag_images(tags=tag_name_to_id, category='valid')
    roboflow_data.tag_images(tags=tag_name_to_id, category='test')
    upload_images(roboflow_data)
