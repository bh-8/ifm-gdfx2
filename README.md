# ifm-gdfx2
Implementation in Forensics and Media Security: Generalizable DeepFake Detection Framework
Initial Test Commit

- build docker image: `./build.sh`
- run framework: `./gdfx2.sh`
## NVIDIA CUDA toolkit setup using Docker
- installation of NVIDIA driver required (check for CUDA version via `nvidia-smi`, CUDA 12.2+ required)
- ```curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list```
- ```sudo apt update && sudo apt install nvidia-container-toolkit -y```
- ```sudo nvidia-ctk runtime configure --runtime=docker```
- ```sudo systemctl restart docker```
