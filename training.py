import os
from dotenv import load_dotenv
import time

from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import CustomBaseModelInfo
from msrest.authentication import ApiKeyCredentials

load_dotenv()

ENDPOINT = os.getenv("VISION_TRAINING_ENDPOINT")
TRAINING_KEY = os.getenv("VISION_TRAINING_KEY")
DOMAIN_ID = os.getenv("DOMAIN_ID")
PROJECT_ID = os.getenv("PROJECT_ID")

credentials = ApiKeyCredentials(in_headers={"Training-key": TRAINING_KEY})
trainer = CustomVisionTrainingClient(ENDPOINT, credentials)

print("Training...")
iteration = trainer.train_project(PROJECT_ID,
                                  custom_base_model_info=CustomBaseModelInfo(project_id=PROJECT_ID,
                                                                             iteration_id='Iteration1'))
while iteration.status != "Completed":
    iteration = trainer.get_iteration(PROJECT_ID, iteration.id)
    print("Training status: " + iteration.status)
    if iteration.training_error_details:
        print("Error details: " + iteration.training_error_details)
        break
    time.sleep(60)
