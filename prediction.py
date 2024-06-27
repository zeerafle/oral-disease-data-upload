import os

from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from dotenv import load_dotenv
from msrest.authentication import ApiKeyCredentials

load_dotenv()

PREDICTION_KEY = os.getenv("VISION_PREDICTION_KEY")
ENDPOINT = os.getenv('VISION_PREDICTION_ENDPOINT')
PROJECT_ID = os.getenv('PROJECT_ID')
published_name = "Iteration1"
base_image_location = os.path.join(os.getenv('DATA_DIR'), 'Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset', 'Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset', 'Data', 'images', 'val')

prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": PREDICTION_KEY})
predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)


# Open the sample image and get back the prediction results.
with open(os.path.join(base_image_location, "(225).jpg"), mode="rb") as test_data:
    results = predictor.detect_image(PROJECT_ID, published_name, test_data)

# Display the results.
for prediction in results.predictions:
    print("\t" + prediction.tag_name + ": {0:.2f}% bbox.left = {1:.2f}, bbox.top = {2:.2f}, bbox.width = {3:.2f}, bbox.height = {4:.2f}".format(prediction.probability * 100, prediction.bounding_box.left, prediction.bounding_box.top, prediction.bounding_box.width, prediction.bounding_box.height))
