# register model

import json
import mlflow
import logging
from src.logger import logging
import os
import dagshub

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "Faizolam"
repo_name = "MLOpsCapstoneProject"
# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------


# Below code block is for local use
# -------------------------------------------------------------------------------------
# mlflow.set_tracking_uri('https://dagshub.com/Faizolam/MLOpsCapstoneProject.mlflow')
# dagshub.init(repo_owner='Faizolam', repo_name='MLOpsCapstoneProject', mlflow=True)
# -------------------------------------------------------------------------------------


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        # Build model URI from the run information and register it
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)

        client = mlflow.tracking.MlflowClient()

        # Optional metadata
        description = model_info.get('description') or model_info.get('model_description')
        tags = model_info.get('tags') or {}
        aliases = model_info.get('aliases')

        # Update description if provided
        if description:
            try:
                client.update_model_version(name=model_name, version=model_version.version, description=str(description))
            except Exception:
                # Older MLflow versions may not support update_model_version; set as tag instead
                client.set_model_version_tag(name=model_name, version=model_version.version, key='description', value=str(description))

        # Set tags (key/value) on the model version
        if isinstance(tags, dict):
            for k, v in tags.items():
                try:
                    client.set_model_version_tag(name=model_name, version=model_version.version, key=str(k), value=str(v))
                except Exception:
                    logging.warning('Failed to set tag %s on model %s:%s', k, model_name, model_version.version)

        # Assign aliases using MLflow's first-class aliases API (recommended over deprecated stages)
        if aliases:
            try:
                # Use MLflow's first-class aliases API to set human-readable aliases for easy model retrieval
                client.set_registered_model_aliases(name=model_name, aliases=str(aliases), version=model_version.version)
                logging.info('aliases "%s" set on model %s version %s', aliases, model_name, model_version.version)
            except Exception as e:
                # Fallback: store aliases as a tag for older MLflow versions
                try:
                    client.set_model_version_tag(name=model_name, version=model_version.version, key='aliases', value=str(aliases))
                    logging.info('aliases stored as tag (older MLflow): %s', aliases)
                except Exception as e2:
                    logging.warning('Failed to set aliases on model %s:%s: %s', model_name, model_version.version, e2)

        logging.debug('Model %s version %s registered. description=%s tags=%s aliases=%s', model_name, model_version.version, description, tags, aliases)
    except Exception as e:
        logging.error('Error during model registration: %s', e)
        raise

def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)
        
        model_name = "my_model" #Creating new registery to keep all new created models(with version) in one registry(new models creates due to new changes in test train prameters and changes in feature, etc)
        # experiment_info.json can provide optional metadata: description (str), tags (dict), aliases (str)
        # These are applied to the registered model version
        register_model(model_name, model_info)
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()

