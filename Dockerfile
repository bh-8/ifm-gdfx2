FROM debian AS gdfx2
    RUN apt update && apt install -y \
        python3 \
        python3-pip \
        python3-venv
    RUN python3 -m venv /home/gdfx2/venv
    RUN . /home/gdfx2/venv/bin/activate && pip3 install \
        torch \
        torchvision \
        torchaudio \
        && deactivate
    WORKDIR /home/gdfx2
    # ENTRYPOINT [""]