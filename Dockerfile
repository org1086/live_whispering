FROM ubuntu:18.04 as base

ENV TZ=Asia/Ho_Chi_Minh
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update
RUN apt-get install -y tzdata

RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get install python3.9 python3.9-dev python3.9-distutils -y
RUN ln -nsf /usr/bin/pip3 /usr/bin/pip && \
    ln -nsf /usr/bin/python3.9 /usr/bin/python && \
    ln -nsf /usr/bin/python3.9 /usr/bin/python3

RUN apt-get install build-essential libssl-dev libffi-dev libxml2-dev libxslt1-dev zlib1g-dev python-pip python3-pip -y 
RUN python3.9 -m pip install --user --upgrade setuptools && \
    python3.9 -m pip install --user --upgrade pip && \
    python3.9 -m pip install --user --upgrade distlib
RUN python3.9 -m pip install --user torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

WORKDIR /app
COPY requirements.txt requirements.txt
RUN python3.9 -m pip install --user -r requirements.txt
COPY . .

CMD [ "python3.9", "-m" , "backend.app_live_whisper", "run", "--host=0.0.0.0", "--port=5000"]
