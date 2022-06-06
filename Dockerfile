FROM python:3.7
WORKDIR /app
COPY ./app /app
COPY requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "15400"] 
