docker build -t read --file docker/DockerfileARM $PWD/docker
docker run -it  -d -v $PWD/:/workspace --net host -p 80:80 --shm-size=64g read bash