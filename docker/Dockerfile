FROM codegram/apex-pytorch1.4-cuda10.1:latest

WORKDIR /workspace

COPY requirements.txt /workspace

RUN pip install -r requirements.txt
RUN pip install wandb spell tensorboard
