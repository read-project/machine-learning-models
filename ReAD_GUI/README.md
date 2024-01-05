# ReAD GUI for machine learning models
Computer vision model to classify images between:
* Photos
* Draws

and to classify Draws with a multi-label classifier between:
* Sections
* Elevations
* Plans
* Others

No text or description is needed.
## For Users
Use following [link](http://150.146.211.35/read_serv/upload) to access the web app.
You can display images classification or download results in JSON and RDF format.
## For Developers
Docker are used to run the interface to the machine learning models.  You will find:
* arm docker (for dev.)
* x86 docker from [meinheld-gunicorn-flask-docker](https://github.com/tiangolo/meinheld-gunicorn-flask-docker/tree/master) (for production)

### Configuration
in docker/Dockfile use:
```
ENV PREFIX="/read_serv"
```
to set the url prefix based on your web server routing configuration (delete if no prefix is required).

### Models weights
To download model weights use following [link](http://150.146.211.35/read_serv/download/models_weights.tar)
### Images datasets
* To download Picture vs Draw classifier dataset use following link: [http://150.146.211.35/read_serv/download/model1_img.tar](http://150.146.211.35/read_serv/download/model1_img.tar) 
* To download Draw classifier (Plans, Sections, Elevations, Other) dataset use following link: [http://150.146.211.35/read_serv/download/model2_img.tar](http://150.146.211.35/read_serv/download/model2_img.tar)
### Model web services
You can check service status by command line:
```
curl -X POST 150.146.211.35:80/read_serv/ping -H "Content-Type: application/json" -H "Accept: application/json" -d '{"ping":0}'
```
Or you can use [Postman](https://www.postman.com/) as client.<br>
Use this endpoint to ping the service http://150.146.211.35/read_serv/ping <br>
and following Body:
```
{"ping":0}
```
if the service is correctly call it will responds:
```
{"response":"ok"}
```
To call models web services you have to:
* convert images in base64 (i.e. using [this](https://www.base64-image.de/) on-line encoder)
* use following endpoint:http://150.146.211.35/read_serv/predict
* in Body put following JSON:
    ```
    {"id":"image_name.jpg",
     "output_type":"JSON",
     "image":"... here put the long base64 encoding ..."
    }
    ```
  you can use
  ```
  "output_type":"RDF"
  ```
  to get RDF output format.
