FROM nvidia/cudagl:9.0-devel-ubuntu16.04

# Install a few libraries to support both EGL and OSMESA options
RUN apt-get update && apt-get install -y wget doxygen curl libjsoncpp-dev libepoxy-dev libglm-dev libosmesa6 libosmesa6-dev libglew-dev libopencv-dev python-opencv python-setuptools python-dev
RUN easy_install pip
RUN pip install backports.functools-lru-cache==1.4 cycler==0.10.0 decorator==4.1.2 matplotlib==2.1.0
RUN pip install networkx==2.0 numpy==1.13.3 olefile==0.44 pandas==0.21.0 Pillow==4.3.0
RUN pip install pyparsing==2.2.0 python-dateutil==2.6.1 pytz==2017.3 PyYAML==3.12
RUN pip install six==1.11.0 subprocess32==3.2.7 torch==1.0.0 torchvision==0.1.9

#install latest cmake
ADD https://cmake.org/files/v3.12/cmake-3.12.2-Linux-x86_64.sh /cmake-3.12.2-Linux-x86_64.sh
RUN mkdir /opt/cmake
RUN sh /cmake-3.12.2-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
RUN cmake --version

ENV PATH /usr/local/lib:/usr/include/python2.7:$PATH
RUN apt-get update && apt-get install -y tmux expect cmake unzip git
RUN git clone https://github.com/mmurray/cvdn.git /opt/MatterSim
RUN cd /opt/MatterSim && git submodule update --init --recursive && mkdir build && cd build && cmake -DEGL_RENDERING=ON -DPYTHON_INCLUDE_DIR=/usr/include/python2.7 -DPYTHON_EXECUTABLE=/usr/bin/python .. 
RUN cd /opt/MatterSim/build && make -j8