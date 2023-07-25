import json
import os

import jinja2
import numpy as np
import cv2
from flask import Flask, jsonify, request, render_template, render_template_string, Response, Blueprint, \
    send_from_directory, current_app

from draw_classif_model import Model, MODEL_FILE, MODELS_FOLDER, ocv_resize_to_rgb

from img_draw_classif_model import Model as Model_img_draw
from img_draw_classif_model import MODEL_FILE as MODEL_FILE_IMG_DRAW
from img_draw_classif_model import MODELS_FOLDER as MODELS_FOLDER_IMG_DRAW
from img_draw_classif_model import LABELS as LABELS_IMG_DRAW
from my_util import system_info
import base64

DEBUG = True   #True
VERBOSE = False #True

RDF_PREFIX_FILE='prefix.txt'
RDF_PREFIX_FOLDER='templates'

res_list=[]     #list of Pred_res object used to store results

#Set Blueprint
#This get the environment variable as prefix for urls
PREFIX = os.getenv('PREFIX', '')
bp = Blueprint('read_app', __name__,  url_prefix=PREFIX) #, template_folder= PREFIX+'/templates', static_folder= PREFIX+'/static', static_url_path= PREFIX+'/static')


model_1 = Model_img_draw(MODELS_FOLDER_IMG_DRAW, MODEL_FILE_IMG_DRAW)       #Model classifing images and draws
model_2 = Model(MODELS_FOLDER, MODEL_FILE, verbose=VERBOSE)                 #Model classifing draws

def base64_predict(encoded_data):
    '''
    Take a base64 image, transform to OpenCV and predict.
    :param encoded_data:
    :return: prediction and level af affidability: 1 is ok, 0 is not affidable
    '''
    # Decode JPEG back into Numpy array
    data = base64.b64decode(encoded_data)           #base64 to byte
    data = np.frombuffer(data, dtype=np.uint8)      #1d array

    img = cv2.imdecode(data, flags=cv2.IMREAD_COLOR)#openCV image

    if VERBOSE:
        print('open cv size:', img.shape)

    ready_img = ocv_resize_to_rgb(img, norm=False, b_w=False, img_height=224, img_width=224, aug_contrast=False)

    if VERBOSE:
        print('image type', type(ready_img), '-shape', ready_img.shape)

    res1=model_1.predict([ready_img])
    if LABELS_IMG_DRAW.index(res1[0][0])==1:
        res2= model_2.predict([ready_img])
    else:
        res2=([[None]],[None])

    return res1 , res2


def to_str():
    #Transform the result list (res_list) in to string
    #Return string
    global res_list
    str_res='{"results": ['
    for i, result in enumerate(res_list):
        str_res+='{"id": "'+result.name+'","type": '
        str_res+=str(result.predA[0])+','
        str_res+= '"content": {"type":'+ str(result.predB[0][0])+',"level": '+str(result.predB[1][0])+'}}'

        if i<len(res_list)-1:
            #Not last element
            str_res+=','

    str_res+=']}'

    str_res= str_res.replace("'",'"')
    str_res = str_res.replace("[None]", '[]')
    str_res = str_res.replace("None", 'null')
    if DEBUG:
        print('json string', str_res)
    return str_res

def to_dic():
    #Transform the result list (res_list) in to dic
    #Return dic
    str_res= to_str()
    return json.loads(str_res)

def to_rdf():
    #Transform the result list (res_list) in to json
    #Return json
    dic_res= to_dic()
    if DEBUG:
        print('Dictionary:', dic_res)
    #load prefixes
    # Load thresholds
    try:
        with open(os.path.join(RDF_PREFIX_FOLDER, RDF_PREFIX_FILE), 'r') as file:
            rdf_out= file.read()
            if DEBUG:
                print('Loaded RDF prefixes from file:\n',rdf_out)
    except:
        print('File', os.path.join(RDF_PREFIX_FOLDER, RDF_PREFIX_FILE), 'not found.\nGo with default prefixes..')
        rdf_out='@prefix a-cd: <https://w3id.org/arco/ontology/construction-description> . \
                 @prefix read: <https://w3id.org/read/> . \
                 @prefix rs: <https://w3id.org/read/resource/> . \
                 @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .'

    #This is the core block for RDF generation
    for res in dic_res['results']:
        if len(res['type']) > 0:
            #is valid picture
            rdf_out += '\nrs:'+str(res['id'])+' a '

            if res['type'][0].lower() == 'photo':
                rdf_out += 'a-cd:PhotographicDocumentation'
            else:
                #Modifica richiesta da Margherita
                #Se contiene solo Others allora e' di tipo read:Drawing e non ha core:hasType
                types_array = res['content']['type']
                if len(types_array) == 1 and types_array[0].lower() == 'others':
                    rdf_out += 'read:Drawing ;'
                else:
                    rdf_out += 'a-cd:GraphicOrCartographicDocumentation ;'
                    for type in res['content']['type']:
                        rdf_out += '\n'+' core:hasType rs:'+str(type)+';'
                rdf_out += '\n' + ' read:predictionAccuracyOnType "' + str(res['content']['level']) + '"'
            rdf_out += '.'
    if DEBUG:
        print(rdf_out)
    return rdf_out

class Pred_Res:
    def __init__(self, img, name, pred1, pred2):
        self.image = img
        self.name = name
        self.predA = pred1
        self.predB = pred2

@bp.route("/ping", methods=["POST"])
def ping():
    payload = json.loads(request.data)
    if DEBUG:
        print(payload)
        system_info()
    return jsonify({"response":"ok"}), 200

class Data:
    def __init__(self, number):
        self.nome = "Fabio"
        self.number = number

@bp.route("/<number>")
def viz(number):
    data = Data(number)
    return render_template("main.html", roba=data)

@bp.route("/")
def base():
    return viz(0)

@bp.route('/upload', methods = ['POST'])
def upload():
    if request.method == 'POST':
        global res_list
        res_list=[]
        # Get the list of files from webpage
        files = request.files.getlist("file")
        # Get output type
        output_type= request.form.get('output_type')
        if VERBOSE:
            print('N. of files:', len(files))
            print('Output type', output_type)

        # Iterate for each file in the files List, and predict
        for file in files:
            if VERBOSE:
                print("Elaborating file", file.filename)
            try:
                # read image file string data and convert to base 64
                b64img= base64.b64encode(file.read())

                if len(b64img) == 0:
                    continue

                #Prediction
                res1, res2 = base64_predict(b64img)
                if VERBOSE:
                    print("Filename: ", file.filename, 'type', type(b64img), res1, res2)
                img= b64img.decode("utf-8")
                res_list.append(Pred_Res(img, file.filename, res1, res2))
            except:
                res_list.append(Pred_Res(None, file.filename, [[None]], ([[None]],[None]) ))

        if output_type == 'img':
                #Display images and results
                return render_template("main.html", results=res_list)

        elif output_type == 'json':
                #Display JSON
                return render_template("json_rdf.html", results=res_list, json_results= to_dic())

        elif output_type == 'rdf':
                # Display RDF
                return render_template("json_rdf.html", results=res_list, rdf_results=to_rdf())

    return render_template_string('PageNotFound {{ errorCode }}', errorCode='404'), 404


@bp.route('/download_json')
def download_json():
    return Response(
        json.dumps(to_dic()),
        mimetype='text/plain',
        headers={'Content-disposition': 'attachment; filename=read_download.json'})

@bp.route('/download_rdf')
def download_rdf():
    return Response(
        to_rdf(),
        mimetype='text/plain',
        headers={'Content-disposition': 'attachment; filename=read_download.rdf'})

@bp.route('/download/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    # Appending app path to upload folder path within app root folder
    downloads = 'downloads'   #os.path.join(current_app.root_path, 'downloads')
    # Returning file from appended path
    if DEBUG:
        msg= 'Download: folder='+downloads+',file='+filename
        print(msg)  #current_app.logger.info(msg)
    return send_from_directory(directory=downloads, path=filename)

@bp.route('/how_to')
def how_to():
    img_path=os.path.join(bp.url_prefix,'static/img')   #os.path.join(app.config['UPLOAD_FOLDER'], 'shovon.jpg')
    return render_template("how_to.html", path_to_img=img_path)

@bp.route("/predict", methods=["POST"])
def predict_img():
    '''
    Use this endpoint to json call:
    {"image": base64 encoded image}
    :return:
        {"res": ...}
    '''
    global res_list
    res_list=[]
    try:
        payload = json.loads(request.data)
        if DEBUG:
            print('Predict the image')

        id= payload['id']

        #Try to set output type
        output_type='json'
        try:
            ot=payload['output_type']
            if ot.lower()=='rdf':
                output_type='rdf'
        except:
            pass

        encoded_data= payload['image']                  #Get byte string from POST
        res1, res2 =base64_predict(encoded_data)        #Decode to bas64 and predict

        img=None
        res_list.append(Pred_Res(img, id, res1, res2))

        if output_type=='json':
            if DEBUG:
                print('Request JSON output...')
            res=jsonify(to_dic())
        else:
            #Produce RDF
            if DEBUG:
                print('Request RDF output...')
            res=to_rdf()
        return res, 200

    except Exception as e :
        if DEBUG:
            print(e)
        return jsonify({"res": "error"}), 500


def ppjson(value, indent=2):
    # prettyprint json for jinja
    return json.dumps(value, indent=indent)

#Filter to use in html rendering
jinja2.filters.FILTERS['ppjson'] = ppjson

#Register app
app = Flask( __name__ )
app.register_blueprint(bp)
if DEBUG:
    print('Url map', app.url_map)

if __name__ == "__main__" :
    app.run( host = '0.0.0.0' ,
             debug = DEBUG ,
             port=int(os.getenv('PORT', 5011)))

