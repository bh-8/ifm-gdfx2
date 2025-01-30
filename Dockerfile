FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04 AS gdfx2
    WORKDIR /home/gdfx2

    # python installation
    RUN apt update && apt install -y \
        python3 \
        python3-pip \
        python3-venv \
        && apt autoremove \
        && apt clean

    # python environment
    RUN python3 -m venv /home/gdfx2/venv
    RUN . /home/gdfx2/venv/bin/activate && pip3 install \
        alive-progress \
        torch \
        torchvision \
        torchaudio \
        && deactivate
    RUN . /home/gdfx2/venv/bin/activate && pip3 install \
        torcheval \
        && deactivate

    # create entrypoint
    RUN echo "#!/bin/bash">/home/gdfx2/gdfx2.sh
    RUN echo ". /home/gdfx2/venv/bin/activate">>/home/gdfx2/gdfx2.sh
    RUN echo "python3 -u /home/gdfx2/gdfx2.py \"\$@\"">>/home/gdfx2/gdfx2.sh
    RUN echo "err_level=\$?">>/home/gdfx2/gdfx2.sh
    RUN echo "deactivate">>/home/gdfx2/gdfx2.sh
    RUN echo "exit \$err_level">>/home/gdfx2/gdfx2.sh
    RUN chmod +x /home/gdfx2/gdfx2.sh

    # copy python source
    COPY ./src /home/gdfx2/
    
    ENTRYPOINT ["/home/gdfx2/gdfx2.sh"]
