import pathlib

UPLOAD_FOLDER = "uploads"
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.absolute()
CLASSIFIER_MODEL_PATH = PROJECT_ROOT / "models" / "document_classifier"