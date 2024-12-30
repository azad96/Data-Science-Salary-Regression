FROM svizor/zoomcamp-model:3.11.5-slim

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "predict.py", "model.bin", "./"]

RUN pip install pipenv && \
    pipenv install --system --deploy

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
