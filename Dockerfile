#Dockerfile,Image,Container
#import base_image
From python:3.7.6

#add all your filename you want to use
ADD Env.py .
ADD utils.py .
ADD main.py .

#to solve cv2 package problem, I add these 2 command
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

#import packages
copy requirements.txt .
RUN pip install -r requirements.txt

#enter filename you want to use
CMD ["python","./main.py"]

#type this in your terminal to build the virtual env
docker build -t sophia .

#type this in your terminal to run the file in line 19
docker run sophia
#(sophia can be replace by any word)