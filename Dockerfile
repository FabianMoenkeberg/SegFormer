# For more information, please refer to https://aka.ms/vscode-docker-python
# FROM python:3.9-slim
FROM nvidia/cuda:12.6.0-base-ubuntu22.04

# RUN nvidia-smi

RUN apt-get update && apt-get install -y python3.10 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \  
    openssh-server \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python3 --version

# configure SSH for communication with Visual Studio 
RUN mkdir -p /var/run/sshd

RUN echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config && \ 
   ssh-keygen -A 

# add user
RUN useradd -m -d /home/dev -s /bin/bash -G sudo dev

# set ssh password for user dev
RUN  echo 'dev:123' | chpasswd

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

ENV NVIDIA_VISIBLE_DEVICES=all


# Set environment variables
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install pip requirements
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

# FROM base as debug
# RUN pip install debugpy

WORKDIR /app
# COPY . /app

# create entrypoint
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
# RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
# USER appuser



# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
# CMD ["python3", "IterativeUnet.py"]
CMD ["bash"]
