docker build -t pytorch .
docker tag pytorch:latest images.borgy.elementai.net/rmst/pytorch:latest
docker push images.borgy.elementai.net/rmst/pytorch:latest

# test with
# sudo docker run -it pytorch bash