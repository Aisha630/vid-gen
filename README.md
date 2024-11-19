# Video Generation

This repository is part of a research project for CS 6303 Topics in LLM, focused on reducing web traffic by enabling video generation on end devices. By generating videos locally on end-user devices through generative models, this project aims to address these issues, reducing data transmission requirements while maintaining video quality.

---

# Set up

For this particular project, we are using the `diffusers-0-27-0` python environment. The requirements are all frozen into the `our_ requirements.txt`.

### Docker container

For this project, we use the image `video-generation`, the container `video-generation-neat`, with the local directory `Video Generation` mounted as a volume

The command to run the contianer is

```bash
docker run -it
    --name video-generation-neat
    -v "/home/iml1/Desktop/Video Generation":/root
    --gpus all
    video_generation
```
