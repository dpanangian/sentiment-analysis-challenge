FROM python:3.7
RUN pip3 install transformers==2.8.0  fastapi uvicorn torch pydantic python-dotenv nltk
RUN python -m nltk.downloader stopwords
RUN mkdir models
RUN pip3 install gdown && \
    gdown --output ./models/model_state_dict.bin --id 1VliP3l8SxcgBN2B4PHwwE-Yn-1k5kzSS

COPY ./app /app
COPY ./config.json /app/config.json
WORKDIR /app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 
