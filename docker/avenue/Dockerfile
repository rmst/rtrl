# syntax=docker/dockerfile:1.0.0-experimental
# (experimental ssh forwarding: https://medium.com/@tonistiigi/build-secrets-and-ssh-forwarding-in-docker-18-09-ae8161d066)

ARG BASE
FROM ${BASE}

WORKDIR /app

ENV GIT_SSH_COMMAND 'ssh -o "StrictHostKeyChecking no"'

# download Avenue assets
ENV AVENUE_ASSETS /app/avenue_assets
RUN mkdir /app/avenue_assets \
  && chmod 777 -R /app/avenue_assets \
  && pip --no-cache-dir install gdown \
  && apt-get update && apt-get install -y --no-install-recommends unzip && apt-get clean && rm -rf /var/lib/apt/lists/*

#RUN mkdir /app/avenue_assets/avenue_follow_car-linux \
#  && gdown -O avenue.zip --id 1eRKQaRxp2dJL9krKviqyecNv5ikFnMrC \
#  && unzip avenue.zip -d /app/avenue_assets/avenue_follow_car-linux \
#  && rm avenue.zip \
#  && chmod 777 -R /app/avenue_assets

ARG AVENUE_REV
RUN --mount=type=ssh git clone git@github.com:elementai/avenue.git avenue \
  && cd avenue \
  && git reset --hard ${AVENUE_REV?} \
  && pip --no-cache-dir install -e .

# download Avenue assets
#RUN mkdir /app/avenue_assets
#ENV AVENUE_ASSETS /app/avenue_assets
#RUN python -c 'import avenue; avenue.download("AvenueCar")'
#RUN chmod 777 -R /app/avenue_assets