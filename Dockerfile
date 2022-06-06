FROM python:3.7
RUN pip3 transformers==2.8.0 install fastapi uvicorn torch pydantic python-dotenv nltk
RUN python -m nltk.downloader stopwords
COPY ./app /app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "15400"] 
