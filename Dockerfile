FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
ENV ANACONDA /opt/anaconda2
ENV CUDA_PATH /usr/local/cuda
ENV PATH ${ANACONDA}/bin:${CUDA_PATH}/bin:$PATH
ENV LD_LIBRARY_PATH ${ANACONDA}/lib:${CUDA_PATH}/bin64:$LD_LIBRARY_PATH
ENV C_INCLUDE_PATH ${CUDA_PATH}/include
RUN apt-get update && apt-get install -y wget build-essential axel imagemagick
RUN wget https://repo.continuum.io/archive/Anaconda2-5.0.1-Linux-x86_64.sh -P /tmp
RUN bash /tmp/Anaconda2-5.0.1-Linux-x86_64.sh -b -p $ANACONDA
RUN rm /tmp/Anaconda2-5.0.1-Linux-x86_64.sh -rf
RUN conda install -y pytorch=0.3.0 torchvision cuda90 -c pytorch
RUN pip install scikit-umfpack
RUN pip install cupy
RUN pip install pynvrtc

