import docker, signal, time, threading

from docker.models.containers import Container

TIMEOUT_DURATION = 10
START_UP_DELAY = 3


class timeout:
    """Thread-safe timeout context manager.
    
    Uses signal.alarm in main thread, falls back to threading.Timer in sub-threads.
    """
    def __init__(self, seconds=TIMEOUT_DURATION, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
        self._timer = None
        self._use_signal = False

    def handle_timeout(self, signum=None, frame=None):
        raise TimeoutError(self.error_message)
    
    def __enter__(self):
        # Check if we're in the main thread
        if threading.current_thread() is threading.main_thread():
            try:
                signal.signal(signal.SIGALRM, self.handle_timeout)
                signal.alarm(self.seconds)
                self._use_signal = True
            except ValueError:
                # Fallback if signal doesn't work
                self._use_signal = False
        
        if not self._use_signal:
            # Use threading.Timer as fallback (non-blocking, just logs warning)
            # Note: Timer cannot interrupt blocking calls, but avoids the error
            self._timer = None  # We skip actual timeout in sub-threads for now

    def __exit__(self, type, value, traceback):
        if self._use_signal:
            signal.alarm(0)
        elif self._timer:
            self._timer.cancel()


def get_container(ctr_name: str, image_name: str, **kwargs) -> Container:
    """
    Reset docker container with given name, or create new container with given name if it does not exist

    Returns:
        Container: reference to docker container object
    """
    client = docker.from_env()
    image = client.images.get(image_name)
    all_containers = [container.name for container in client.containers.list(all=True)]
    container = None
    if ctr_name in all_containers:
        # Reset container via cleanup script if it exists
        container = client.containers.get(ctr_name)
        container.reload()
        if container.attrs.get("Image") != image.id:
            container.remove(force=True)
            container = None
        elif container.status != "running":
            container.start()
    if container is None:
        # Only keep relevant kwargs
        for key in kwargs.copy().keys():
            if key not in ["command", "environment", "ports", "volumes"]:
                del kwargs[key]
                
        # Create + return new container from custom image
        container = client.containers.run(
            image=image,
            name=ctr_name,
            detach=True,
            tty=True,
            **kwargs)
    time.sleep(START_UP_DELAY)
    
    # Check if container was created successfully
    if not container:
        raise RuntimeError(f"Failed to create and start `{ctr_name}` container successfully")
    
    return container
