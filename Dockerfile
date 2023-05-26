# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Update and install gcc and g++
RUN apt-get update && apt-get install -y gcc g++

# Add the necessary dependencies for tesserocr
RUN apt-get update && \
    apt-get install -y tesseract-ocr libtesseract-dev libleptonica-dev pkg-config && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "main.py"]
