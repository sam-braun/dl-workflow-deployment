FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
WORKDIR /opt/pytorch-mnist
ADD mnist.py /opt/pytorch-mnist
ADD requirements.txt /opt/pytorch-mnist
RUN pip install --no-cache-dir -r requirements.txt
ENTRYPOINT ["python", "mnist.py"]