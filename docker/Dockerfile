# Use Miniconda for easier environment management
FROM continuumio/miniconda3:latest

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

# Copy environment definition first to leverage Docker layer caching
COPY config/environment.yml /tmp/environment.yml

# Create the conda environment
RUN conda env create -f /tmp/environment.yml

# Activate the environment
SHELL ["/bin/bash", "-c"]
ENV PATH /opt/conda/envs/pogo-pvp-assistant/bin:$PATH

# Copy the current directory contents into the container at /app
COPY . /app

# Make port 5037 available
EXPOSE 5037

# Define environment variable for ADB
ENV ANDROID_ADB_SERVER_ADDRESS=host.docker.internal
ENV PYTHONUNBUFFERED=1

# Run the application using the conda environment
CMD ["conda", "run", "-n", "pogo-pvp-assistant", "python", "-m", "src.main"]
