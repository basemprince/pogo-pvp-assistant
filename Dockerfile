# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container to /app
WORKDIR /app

# Update and install necessary dependencies
RUN apt-get update && \
    apt-get install --fix-missing -y \
        gcc g++ bash tesseract-ocr \
        libtesseract-dev libleptonica-dev \
        pkg-config ffmpeg libsm6 libxext6 tk \
        net-tools inetutils-ping inetutils-telnet \
        android-sdk-platform-tools-common && \
    rm -rf /var/lib/apt/lists/* 

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r docker-req.txt

# Make port 80 available to the world outside this container
EXPOSE 5037


# Define environment variable for ADB
ENV ANDROID_ADB_SERVER_ADDRESS=host.docker.internal
ENV PYHTONUNBUFFERED=1

# Run bash when the container launches
# CMD ["bash"]

CMD ["python", "main.py"]