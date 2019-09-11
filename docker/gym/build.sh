docker build -t gym .
docker tag gym:latest images.borgy.elementai.net/rmst/gym:latest
docker push images.borgy.elementai.net/rmst/gym:latest

# test with docker run -it images.borgy.elementai.net/rmst/gym:latest