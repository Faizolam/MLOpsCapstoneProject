FROM python:3.10-slim

WORKDIR /app

# Stable (rarely changes)
# 1 Copy only requirements first
COPY flask_app/requirements.txt /app/
# 2 Install dependencies (cached unless requirements.txt changes)
RUN pip install --no-cache-dir -r requirements.txt
# 3 Download nltk data (cached)
RUN python -m nltk.downloader stopwords wordnet

# Frequently changes
# 4 Copy application code
COPY flask_app/ /app/
# 5 Copy model file
COPY models/vectorizer.pkl /app/models/vectorizer.pkl

EXPOSE 5000

#local
CMD ["python", "app.py"]  

#Prod
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]