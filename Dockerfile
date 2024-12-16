# Use a Miniconda base image with Python 3.8
FROM continuumio/miniconda3

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container to /VideoReconstruction
WORKDIR /VideoReconstruction

# Copy all files from the current directory into the container's working directory
COPY . /VideoReconstruction

# Create the Conda environment from the environment.yml file
RUN conda env create -f environment.yml
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN conda env create -f visil_env.yml

# Start in interactive mode
CMD ["/bin/bash"]