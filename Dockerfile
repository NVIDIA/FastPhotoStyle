FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
# Set anaconda path
ENV ANACONDA /opt/anaconda
ENV CUDA_PATH /usr/local/cuda
ENV PATH ${ANACONDA}/bin:${CUDA_PATH}/bin:$PATH
ENV LD_LIBRARY_PATH ${ANACONDA}/lib:${CUDA_PATH}/bin64:$LD_LIBRARY_PATH
# Download anaconda and install it
RUN apt-get update && apt-get install -y wget build-essential
RUN apt-get update && apt-get install -y libopencv-dev python-opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev
RUN wget https://repo.continuum.io/archive/Anaconda2-5.0.1-Linux-x86_64.sh -P /tmp
RUN bash /tmp/Anaconda2-5.0.1-Linux-x86_64.sh -b -p $ANACONDA
RUN rm /tmp/Anaconda2-5.0.1-Linux-x86_64.sh -rf
RUN conda install -y pytorch torchvision cuda90 -c pytorch
RUN conda install -y -c menpo opencv3
#RUN conda install -y -c anaconda pip 
RUN pip install scikit-umfpack
RUN pip install -U setuptools
RUN pip install cupy
RUN pip install pynvrtc
RUN apt-get install -y axel
RUN apt-get install -y imagemagick
ENV C_INCLUDE_PATH ${CUDA_PATH}/include
