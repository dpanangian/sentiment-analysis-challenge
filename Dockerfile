FROM python:3.7
RUN apt update
RUN apt install git-lfs

RUN pip3 install transformers==2.8.0  fastapi uvicorn torch pydantic python-dotenv nltk scikit-learn pickle-mixin
RUN python -m nltk.downloader stopwords

COPY ./api src/api
COPY ./config.json /src/config.json
WORKDIR /src
RUN mkdir models
RUN pip3 install gdown && \
    gdown https://drive.google.com/drive/folders/1PvOClskengpMakFUyQFqKELb4ipSpOtL?usp=sharing -O /models --folder

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"] 
