# Docker
We use docker in WSL to run tensorflow version 2.17, which is not supported for Windows anymore. 

## With Docker and VS Code
### Setup Instructions
Important steps, some are already done, e.g. tensorflow[and-cuda]==2.17.0
* Install WSL 2 and VS Code
* Install [Cuda Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-the-nvidia-container-toolkit)
* Install [Cuda-Driver](https://developer.nvidia.com/cuda-12-9-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11) (probably Version 12.9)
* Maybe Install [Cudnn-cuda-12](https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local) or cudnn-cuda-11 depending on coda version
* Use tensorflow[and-cuda]==2.17.0 instead of tensorflow or tensorflow-gpu (already defined in Dockerfile)
* General Infos [Cuda and Docker ](https://docs.nvidia.com/ai-enterprise/deployment/vmware/latest/docker.html)
* Install [cuda-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-the-nvidia-container-toolkit)
### Create/Run Docker Container with GPU support
* Start Docker `sudo service docker start`
* Create Docker container from Dockerfile `docker buildx build -t segformer_image .`
* Create and Run Docker container once `docker run --gpus all --name segFormer -p 2224:24  -it -v /mnt/c/Users/fmo/sourcePrivate/SegFormer:/app segformer_image tail -f /dev/null`
  Later you can just start the container and `Attatch Visual Studio`
* Debug Docker container ``
  
### Definitions to use Docker in VS Code
Important files are 
* `Dockerfile`
* `.vscode/launch.json` -> Launch tasks that can be run directly with `Play`-Button. Important `Python: Run current File inside Docker`.
* `.vscode/tasks.json` -> Definition of single tasks, e.g., build, run (command, file, shell)
* `.devcontainer/devcontainer.json` (this might be not used or wrong)\
	-> ALK knows how to use devcontainers. Probably part of the `docker run ...` command needs to be in there.

## Run and Debug with VS Code
Note: It is important to use `buildx` not `build`.

### Run it in VS Code and attach to it
* Build manually docker container in terminal: `docker buildx build -t segformer_image .`
* Run and Create docker container `docker run --gpus all --name segFormer -p 2224:24  -it -v /mnt/c/Users/fmo/sourcePrivate/SegFormer:/app segformer_image tail -f /dev/null`
* Attach VS Code to running Container called `segFormer`
* Install "Python Debugger" VS Code Extension (if not already installed)
* Use `Python: Run current File inside Docker` to run or debug current file.

### Start Docker Container and attach VS Code to running container.
* Build manually docker container in terminal: `docker buildx build -t segformer_image .`
* Start Container in the Container-Field.
* Right-Click on running container and Attach VS Code to running container.
* Debug and run scripts with launch-json "Python: Run current File inside Docker".

### Run it in VS Code manually
* Build manually docker container in terminal: `docker buildx build -t segformer_image .`
* Comment "Wait for client from debugpy"
* Run it manually in terminal: `docker run --gpus all -v /mnt/c/Users/fmo/sourcePrivate/SegFormer:/app segformer_image`

### Debug it with VS Code

* Build manually docker container in terminal: `docker buildx build -t segformer_image .`
* Uncomment "Wait for client from debugpy"
* Start "Terminal -> Run Task..." the task "docker-run: prebuild"
* In "Run and Debug" run "Docker: Python - Attach Debug"
* debug it ...