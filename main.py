import subprocess
import os
import zipfile
from dotenv import load_dotenv

from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import CustomVisionErrorException, ImageFileCreateBatch
from msrest.authentication import ApiKeyCredentials

from oral.labels import Labels


load_dotenv()

ENDPOINT = os.getenv("VISION_TRAINING_ENDPOINT")
TRAINING_KEY = os.getenv("VISION_TRAINING_KEY")
DOMAIN_ID = os.getenv("DOMAIN_ID")
PROJECT_ID = os.getenv("PROJECT_ID")


def download_dataset():
    command = "kaggle datasets download -d salmansajid05/oral-diseases"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()
    print("Command executed, return code:", process.returncode)


def extract_dataset(data_dir: str):
    with zipfile.ZipFile("oral-diseases.zip", "r") as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(os.path.join('data', 'Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset',
                           'Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset', 'Data', 'labels',
                           'train', 'labels.txt'))


if __name__ == "__main__":
    if not os.path.exists("oral-diseases.zip"):
        download_dataset()
        extract_dataset(os.getenv("DATA_DIR"))

    print("Dataset downloaded and extracted")

    credentials = ApiKeyCredentials(in_headers={"Training-key": TRAINING_KEY})
    trainer = CustomVisionTrainingClient(ENDPOINT, credentials)

    # tagging images
    labels = Labels()

    tags = []
    for class_name in labels.classes:
        try:
            tags.append(trainer.create_tag(PROJECT_ID, class_name))
        except CustomVisionErrorException:
            print(f"Tag {class_name} already exists")
            continue

    if not tags:
        print("Tags already exist")
        tags = trainer.get_tags(PROJECT_ID)

    tag_name_to_id = {tag.name: tag.id for tag in tags}

    labels.tag_images(tags=tag_name_to_id)

    # upload images
    # slice the list into batches of size 64
    for i in range(0, len(labels.tagged_images_with_regions), 64):
        print(i)
        batch = labels.tagged_images_with_regions[i:i + 64]
        upload_result = trainer.create_images_from_files(PROJECT_ID, ImageFileCreateBatch(images=batch))
        if not upload_result.is_batch_successful:
            print("Image batch upload failed")
            for image in upload_result.images:
                print("Image status: ", image.status)
