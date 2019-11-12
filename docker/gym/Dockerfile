ARG BASE
FROM ${BASE}

# mujoco-py requirements https://github.com/openai/mujoco-py/blob/master/Dockerfile
RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    unzip \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf

WORKDIR /app

RUN mkdir -p .mujoco \
    && wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d .mujoco \
    && mv .mujoco/mujoco200_linux .mujoco/mujoco200 \
    && rm mujoco.zip

# will compile even without a valid mjkey
ARG MJ_KEY=""
RUN echo "$MJ_KEY" > .mujoco/mjkey.txt

ENV LD_LIBRARY_PATH /app/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
ENV MUJOCO_PY_MJKEY_PATH /app/.mujoco/mjkey.txt
ENV MUJOCO_PY_MUJOCO_PATH /app/.mujoco/mujoco200

ARG GYM_REV=c33cfd8b2cc8cac6c346bc2182cd568ef33b8821

ARG GYM_FEATURES='[mujoco]'
RUN git clone https://github.com/openai/gym \
 && cd gym \
 && git reset --hard $GYM_REV \
 && pip --no-cache-dir install -e ."${GYM_FEATURES}"

# we need to change the permissions of mujoco_py/generated because mujoco-py will fail if it can't modifiy this directory
RUN printf "\
try: import mujoco_py, os \n\
except: exit() \n\
p = os.path.join(os.path.dirname(mujoco_py.__file__), 'generated') \n\
print(p) \n\
os.chmod(p, 0o777) \n" | python