# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory INSIDE the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
# COPY . .
COPY . .

# Install any needed packages specified in requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y \
    && pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

EXPOSE 5000

# Run app.py when the container launches
CMD ["python", "app.py"]