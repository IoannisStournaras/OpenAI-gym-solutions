FROM docker.io/library/amazonlinux:2.0.20200722.0 as base
ARG UNAME=istournaras
ARG UID=1001
ARG GID=1001

WORKDIR /app

# Basic Configuration
RUN yum -y update \
    && yum install -y shadow-utils python3 python3-devel python3-pip gcc-c++ gcc-gfortran swig cmake xvfb ffmpeg which \
    && yum clean all \
    && python3 -m pip install --upgrade pip wheel \
    && groupadd -g ${GID} -o ${UNAME} \
    && useradd -ms /bin/bash -u ${UID} -g ${GID} -d /app ${UNAME} \
    && mkdir -p /data/pretrained /data/logs \
    && chown -R ${UNAME}: /app \
    && chown -R ${UNAME}: /data

# Required packages for rendering the screen
RUN yum -y install libglvnd libgl libXtst libX11 libglvnd-devel mesa-libGL mesa-libGL-devel freeglut freeglut-devel \
    libXrender libSM libXext mesa-dri-drivers fontconfig-devel drm-utils

VOLUME /data/

ENV USER ${UNAME}
ENV PATH $PATH:/app/.local/bin
#USER ${UNAME}

COPY requirements.txt ./
RUN python3 -m pip install --user -r requirements.txt

FROM base as reinforcement
COPY --chown=${USER} reinforcement ./reinforcement
ENV PYTHONPATH "${PYTHONPATH}:/app/supplydemand"
CMD ["sh", "-c"]