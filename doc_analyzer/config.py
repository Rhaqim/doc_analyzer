import pathlib

UPLOAD_FOLDER = "uploads"
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.absolute()
CLASSIFIER_MODEL_PATH = PROJECT_ROOT / "models" / "document_classifier"
DATABASE_CONN = "db"
DATABASE_PORT = 5432
DATABASE_NAME = "postgres"
DATABASE_USER = "postgres"
DATABASE_PASSWORD = "postgres"
DATABASE_URI = f"postgresql://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_CONN}:{DATABASE_PORT}/{DATABASE_NAME}"
