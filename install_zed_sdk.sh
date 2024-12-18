#!/bin/bash
if [ -f /usr/local/zed/lib/libsl_zed.so ]; then
  echo already installed zed sdk ; exit
fi

apt-get update
apt install sudo
apt install -y zip
apt install zstd
export ZED_SDK_INSTALLER=ZED_SDK_Tegra_L4T35.3_v4.1.0.zstd.run
wget --quiet -O ${ZED_SDK_INSTALLER} https://download.stereolabs.com/zedsdk/4.1/l4t35.2/jetsons
chmod +x ${ZED_SDK_INSTALLER} && ./${ZED_SDK_INSTALLER} -- silent
