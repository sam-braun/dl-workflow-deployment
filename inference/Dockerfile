FROM python:3.8-slim
WORKDIR /app
COPY inference.py /app
COPY model.pth /app  # assuming the model is named model.pth
RUN pip install torch flask
EXPOSE 5000
CMD ["python", "inference.py"]