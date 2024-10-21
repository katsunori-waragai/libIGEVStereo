.PHONY: reformat
reformat:
	black */*.py

.PHONY: test
test:
	cd test && pytest test*.py

.PHONY: install
install:
	python3 -m pip install .

.PHONY: whl
whl:
	apt install -y python3.8-venv
	python3 -m venv venv
	. venv/bin/activate
	python3 -m pip install build
	python3 -m build

.PHONY: download
download:
	mkdir -p ./stereoigev/models/ && cd ./stereoigev/models/ &&  \
	gdown --fuzzy https://drive.google.com/file/d/16e9NR_RfzFdYT5mPaGwpjccplCi82C2e/view?usp=drive_link

.PHONY: install_zed_sdk
install_zed_sdk:
	apt-get update
	apt install sudo
	apt install -y zip
	apt install zstd
	ZED_SDK_INSTALLER=ZED_SDK_Tegra_L4T35.3_v4.1.0.zstd.run
	wget --quiet -O ${ZED_SDK_INSTALLER} https://download.stereolabs.com/zedsdk/4.1/l4t35.2/jetsons
	chmod +x ${ZED_SDK_INSTALLER} && ./${ZED_SDK_INSTALLER} -- silent
