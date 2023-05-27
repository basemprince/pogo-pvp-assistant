# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container to /app
WORKDIR /app

# Update and install gcc and g++
RUN apt-get update && apt-get install -y gcc g++ 
# RUN apt-get update && apt-get install -y net-tools netcat adb
RUN apt-get update &&  apt-get install -y tesseract-ocr libtesseract-dev libleptonica-dev pkg-config ffmpeg libsm6 libxext6 tk


# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r docker-req.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV PYHTONUNBUFFERED=1

# Run app.py when the container launches
CMD ["python", "main.py"]
