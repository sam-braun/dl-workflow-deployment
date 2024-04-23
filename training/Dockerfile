FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
# FROM docker.io/kubeflowkatib/pytorch-mnist:v1beta1-45c5727
WORKDIR /opt/pytorch-mnist
ADD mnist.py /opt/pytorch-mnist
ADD requirements.txt /opt/pytorch-mnist
RUN pip install --no-cache-dir -r requirements.txt
ENTRYPOINT ["python", "mnist.py"]

# # We need to use the nvcr.io/nvidia/pytorch image as a base image to support both linux/amd64 and linux_arm64 platforms.
# # PyTorch=2.2.0, cuda=12.3.2
# # Ref: https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-01.html#rel-24-01
# FROM nvcr.io/nvidia/pytorch:24.01-py3

# RUN pip install tensorboardX==2.6.2
# RUN mkdir -p /opt/mnist

# WORKDIR /opt/mnist/src
# ADD mnist.py /opt/mnist/src/mnist.py

# RUN chgrp -R 0 /opt/mnist \
#     && chmod -R g+rwX /opt/mnist

# ENTRYPOINT ["python", "/opt/mnist/src/mnist.py"]

# FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
# WORKDIR /app
# ADD mnist.py /app
# ADD requirements.txt /app
# RUN pip install --no-cache-dir -r requirements.txt
# ENTRYPOINT ["python", "mnist.py"]

# ADD examples/v1beta1/pytorch-mnist /opt/pytorch-mnist # buildkit
# WORKDIR /opt/pytorch-mnist
# RUN /bin/sh -c mkdir /katib
# RUN /bin/sh -c chgrp -R
# ENTRYPOINT ["python3" "/opt/pytorch-mnist/mnist.py"]