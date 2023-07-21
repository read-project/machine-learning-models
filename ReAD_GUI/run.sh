##docker build -t read --file docker/Dockerfile $PWD/docker
#docker build -t read ./docker
docker build -f docker/Dockerfile -t read .
docker run -d -p 83:8080 -e PORT="8080" -e APP_MODULE="server:app" read
