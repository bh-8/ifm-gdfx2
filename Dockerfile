# FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 AS gdfx2
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04 AS gdfx2
    WORKDIR /home/gdfx2
    RUN apt update && apt install -y \
        python3 \
        python3-pip \
        python3-venv \
        && apt autoremove \
        && apt clean
    RUN python3 -m venv /home/gdfx2/venv
    RUN . /home/gdfx2/venv/bin/activate && pip3 install \
        torch \
        torchvision \
        torchaudio \
        && deactivate
    # ENTRYPOINT [""]
