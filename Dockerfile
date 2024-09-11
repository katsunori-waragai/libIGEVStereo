FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

# for depth anything
RUN apt-get update
RUN apt install sudo
RUN apt install -y zip
RUN apt install -y build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
RUN apt install -y libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev
RUN apt install -y libv4l-dev v4l-utils qv4l2
RUN apt install -y curl
RUN apt install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
RUN apt install zstd
# RUN python3 -m pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
# only for development
RUN apt update && apt install -y eog nano
RUN apt install -y meshlab

ENV ZED_SDK_INSTALLER=ZED_SDK_Tegra_L4T35.3_v4.1.0.zstd.run
RUN wget --quiet -O ${ZED_SDK_INSTALLER} https://download.stereolabs.com/zedsdk/4.1/l4t35.2/jetsons
RUN chmod +x ${ZED_SDK_INSTALLER} && ./${ZED_SDK_INSTALLER} -- silent


# for depth anything
RUN cd /root && git clone https://github.com/gangweiX/IGEV.git
RUN cd /root/IGEV
WORKDIR /root/IGEV
COPY *.py /root/IGEV/IGEV-Stereo/
RUN python3 -m pip install gdown
RUN mkdir -p /root/IGEV/IGEV-Stereo/pretrained_models/sceneflow/; cd /root/IGEV/IGEV-Stereo/pretrained_models/sceneflow/ ; gdown --fuzzy https://drive.google.com/file/d/16e9NR_RfzFdYT5mPaGwpjccplCi82C2e/view?usp=drive_link
COPY pyproject.toml ./
RUN python3 -m pip install .[dev]
ENV LD_PRELOAD=/usr/local/lib/python3.8/dist-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0
WORKDIR /root/IGEV/IGEV-Stereo
COPY cap_and_stereo.sh ./
