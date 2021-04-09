FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

WORKDIR /app

COPY requirements.txt ./

RUN pip install -r requirements.txt
RUN apt update 
RUN apt install -y ffmpeg
COPY ./ ./
EXPOSE 6677

CMD ["python" , "api.py"]
