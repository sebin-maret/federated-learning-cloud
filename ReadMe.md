<!-- Federated Learning based Medical Image Classifier -->
## Federated Learning based Medical Image Classifier


The project is built for the course CS6905. This project consists of a federated learning system that trains a medical image classifer using federated learning algorithm.
The project folder consists of the server and the client code that needs to run separately for the training to take place.
The client has access to the dataset used for the training while the server can be deployed on aly cloud based platforms.

The server can be deployed as a containerized service using the Dockerfile given in the repository.

For containerizing the server code, run the following command
```shell
docker build -t <image-tag> . && docker push  <image-tag>
```

The pushed docker image can then be deployed on any containerization technology. We have used AWS ECS for deploying the server.
The server should be deployed and a loadbalancer IP needs to be configured which is configured in the pyproject.toml file. This settings file is used to 
initiate the training process and uses the IP in here to connect with the server.

The clients can be started using 
```shell
flower-supernode \                  
     --insecure \
     --superlink medportal-aggregator-lb-19961353441ae800.elb.us-east-2.amazonaws.com:9092 \
     --clientappio-api-address 127.0.0.1:9095 --node-config "client-id='client-2'"
```


And the training can be initiated using 

```shell
flwr run . cloud-deployment --stream
```
