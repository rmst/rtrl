# syntax=docker/dockerfile:1.0.0-experimental
# (experimental ssh forwarding: https://medium.com/@tonistiigi/build-secrets-and-ssh-forwarding-in-docker-18-09-ae8161d066)

ARG BASE
FROM ${BASE}

WORKDIR /app

ENV GIT_SSH_COMMAND 'ssh -o "StrictHostKeyChecking no"'

#ARG MLAGENTS_REV
#RUN --mount=type=ssh git clone git@github.com:rmst/ml-agents-frozen.git ml-agents \
#  && cd ml-agents \
#  && git reset --hard $MLAGENTS_REV \
#  && pip --no-cache-dir install Pillow \
#  && pip --no-cache-dir install -e ./ml-agents \
#  && pip --no-cache-dir install -e ./gym-unity

ARG AVENUE_REV
RUN --mount=type=ssh git clone git@github.com:ElementAI/Avenue.git avenue \
  && cd avenue \
  && git reset --hard $AVENUE_REV \
  && pip --no-cache-dir install -e .

# download Avenue assets
RUN mkdir /app/avenue_assets
ENV AVENUE_ASSETS /app/avenue_assets
RUN python -c 'import avenue; avenue.download("AvenueCar")'
RUN chmod 777 -R /app/avenue_assets