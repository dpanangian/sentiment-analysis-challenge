FROM python:3.7
RUN apt-get install git-lfs
RUN pip3 install transformers==2.8.0  fastapi uvicorn torch pydantic python-dotenv nltk
RUN python -m nltk.downloader stopwords

COPY ./app src/app
COPY ./config.json /src/config.json
WORKDIR /src
RUN mkdir models
RUN pip3 install gdown && \
    gdown --output ./models/model_state_dict.bin --id 1VliP3l8SxcgBN2B4PHwwE-Yn-1k5kzSS
RUN mkdir bert
RUN git clone https://huggingface.co/bert-base-uncased ./bert/
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "15400"] 
