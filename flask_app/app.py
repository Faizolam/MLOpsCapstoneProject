from flask import Flask, render_template, request
import mlflow
import pickle
import os
import pandas as pd
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import time
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import re
import dagshub
from pathlib import Path

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

def lemmatization(text):
    """Lemmatize the text."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    """Remove numbers from the text."""
    text = ''.join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    """Convert text to lower case."""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    """Remove sentences with less than 3 words."""
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)

    return text

# Below code block is for local use
# -------------------------------------------------------------------------------------
# mlflow.set_tracking_uri('https://dagshub.com/Faizolam/MLOpsCapstoneProject.mlflow')
# dagshub.init(repo_owner='Faizolam', repo_name='MLOpsCapstoneProject', mlflow=True)
# -------------------------------------------------------------------------------------

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

# Initialize Flask app
app = Flask(__name__)

# from prometheus_client import CollectorRegistry

# Create a custom registry
registry = CollectorRegistry()

# Define your custom metrics using this registry
REQUEST_COUNT = Counter(
    "app_request_count", "Total number of requests to the app", ["method", "endpoint"], registry=registry
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Latency of requests in seconds", ["endpoint"], registry=registry
)
PREDICTION_COUNT = Counter(
    "model_prediction_count", "Count of predictions for each class", ["prediction"], registry=registry
) #Num of positive or negative reviews(eg: let's say you are hiting my app 4 time and those 4 request you got 2 +ve and 2 -ve, so prediction count tell us on an avarage in 1 min how many +ve or -ve review we are getting )

# ------------------------------------------------------------------------------------------
# Model and vectorizer setup(original logic)
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent

model_name = "my_model"

def get_latest_model_version(model_name):
    """Make a connection to MLflow and return the latest model version."""
    client = mlflow.MlflowClient()
    # Use search_model_versions to retrieve all versions; fall back to get_latest_versions for older MLflow
    try:
        versions = list(client.search_model_versions(f"name='{model_name}'"))
    except Exception:
        versions = client.get_latest_versions(model_name) or client.get_latest_versions(model_name, stages=["None"]) or []

    if not versions:
        return None

    def stage_of(v):
        return (getattr(v, 'current_stage', '') or '').lower()

    # Select the highest numeric version by default (most recent registration)
    latest = max(versions, key=lambda v: int(v.version))
    print(f"get_latest_model_version: selected version {latest.version} (stage={(getattr(latest,'current_stage','') or '')})")
    return latest.version


def resolve_repo_path(*parts):
    return REPO_ROOT.joinpath(*parts)

model_version = get_latest_model_version(model_name)
if model_version is None:
    raise RuntimeError(f"No model versions found for '{model_name}'.")

model_uri = f'models:/{model_name}/{model_version}'
print(f"Fetching model from: {model_uri}")
model = mlflow.pyfunc.load_model(model_uri)

vectorizer_path = resolve_repo_path('models', 'vectorizer.pkl')
print(f"Loading vectorizer from: {vectorizer_path}")
if not vectorizer_path.exists():
    raise FileNotFoundError(f"Vectorizer file not found at {vectorizer_path}")

with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

# Routes
@app.route("/")
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    response = render_template("index.html", result=None)
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response

@app.route("/predict", methods=["POST"])
def predict():
    print("========== PREDICT REQUEST RECEIVED ==========", flush=True)
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()

    try:
        text = request.form.get("text", "")
        print(f"Received text: {text[:100]}", flush=True)
        
        # Clean text
        text = normalize_text(text)
        print(f"Normalized text: {text[:100]}", flush=True)
        # Convert to features(transforming text ot vector and making it in numerical features so that model can understandd.)
        features = vectorizer.transform([text])
        print(f"Features shape: {features.shape}", flush=True)
        
        features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])
        print(f"Features DataFrame shape: {features_df.shape}", flush=True)

        # Predict
        result = model.predict(features_df)
        print(f"Model result type: {type(result)}, value: {result}", flush=True)
        
        prediction = result[0]
        print(f"Prediction: {prediction}", flush=True)

        # Increment prediction count metric
        PREDICTION_COUNT.labels(prediction=str(prediction)).inc()

        # Measure latency
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

        return render_template("index.html", result=prediction)
    except Exception as e:
        print(f"ERROR in predict: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", 500

@app.route("/metrics", methods=["GET"])
def metrics():
    """Expose only custom Prometheus metrics."""
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    # app.run(debug=True) # for local use
    app.run(host="0.0.0.0", port=5000)  # Accessible from outside Docker
