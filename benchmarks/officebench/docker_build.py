"""Docker build utilities for OfficeBench."""
import os
import time

import docker

TRUTHY_VALUES = {"1", "true", "yes", "y"}


def should_force_rebuild():
    return os.getenv("OFFICEBENCH_REBUILD_DOCKER", "").lower() in TRUTHY_VALUES


def build_docker(docker_image_name, dockerfile_path):
    """
    Build the docker image for the InterCode Bash environment. If the image already exists, do nothing.
    """
    client = docker.from_env()
    available_images = [y for x in client.images.list() for y in x.tags]
    image_tag = f"{docker_image_name}:latest"
    if image_tag in available_images and not should_force_rebuild():
        print(f"`{image_tag}` already exists. Set OFFICEBENCH_REBUILD_DOCKER=true to rebuild it.")
        return
    print(f"Building `{image_tag}`...")
    
    client.images.build(
        path='./',
        dockerfile=dockerfile_path,
        tag=docker_image_name,
        rm=True,
        nocache=True
    )

    # Give some time for Bash server to start
    print("✓ Bash Docker image built successfully. " + \
          "Waiting for 5 seconds for Bash container to start...\n" + \
          "If you encounter an error, run `docker ps --all` and check if `{docker_image_name}` containers were created. " + \
          "Container start up time varies by machine.")
    time.sleep(5)
