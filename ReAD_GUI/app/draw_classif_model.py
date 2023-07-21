import glob
import os
import pickle

import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications import ResNet50, VGG16
from keras.optimizers import SGD



"""# Init"""
VERBOSE= True

batch_size = 16
img_height = 224
img_width = 224

LABELS = ['Plan', 'Elevation', 'Section', 'Others']

DATA_FOLDER = 'test'  # This is the image and metadata folder
MODELS_FOLDER = 'models'  # This is where you want to store trained models

LS_FILE = 'label-studio_export.json'  # Label Studio export file (metadata file)
MODEL_FILE = 'clf_NN_multi_2.h5'

THRESHOLD_FILE='clf_NN_multi_thresholds.pk'

NORM_IMAGE = False  # if True than normalise between [0,1]
# WARNING! All images must have numpy's dtype uint8. Values are expected to be in
# range 0-255
BW_IMAGE = False  # if True than images are transformed in black and white

print('Data folder:', DATA_FOLDER)



def ocv_resize_to_rgb(img, norm=False, b_w=False, img_height=224, img_width=224, aug_contrast=False):
  '''
  Take an OpenCV (BGR) image, resize it and return an RGB numpy array
  :param img:
  :param norm:
  :param b_w:
  :param img_height:
  :param img_width:
  :param aug_contrast:
  :return: RGB numpy array
  '''
  #Use OpenCV

  if aug_contrast:
    #Augment the contrast of image
    # converting to LAB color space
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))
    # Converting image from LAB Color model to BGR color spcae
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

  #Interpolate
  img= cv2.resize(img, (img_width, img_height) , interpolation= cv2.INTER_AREA) # INTER_CUBIC )
  img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img= np.asarray(img)
  if norm:
    img=img/255
  # Resize the image to the desired size

  return img


def load_and_resize(file_path, norm= False, b_w= False, img_height=224, img_width=224, aug_contrast=False, verbose=False):
  # Load the raw data from the file as a string
  if verbose:
      print(file_path)
  try:
    img = cv2.imread(file_path)
    img = ocv_resize_to_rgb(img, norm, b_w, img_height, img_width, aug_contrast)
    return img
  except:
    return None
"""# Model definition"""

def define_nn_clf(pretrain_mod='res',
                  pretrain_trainable=False,
                  use_rescaling=False,
                  n_layers=1,
                  layer_dim=128,
                  layer_reducer=2,
                  use_dropout=True,
                  dropout=0.3,
                  use_norm=False,
                  multi_label=False,
                  activation='tanh',
                  opt='rmsprop',
                  metrics=['accuracy'],
                  output_shape=1,
                  verbose=False):
    # Initialize the Pretrained Model
    if pretrain_mod == 'res':
        feature_extractor = ResNet50(weights='imagenet',
                                     input_shape=(224, 224, 3),
                                     include_top=False)
    else:
        feature_extractor = VGG16(weights='imagenet',
                                  input_shape=(224, 224, 3),
                                  include_top=False)

    # Set this parameter to make sure it's not being trained
    feature_extractor.trainable = pretrain_trainable

    # Set the input layer
    input_ = tf.keras.Input(shape=(224, 224, 3))

    # Rescaling
    if use_rescaling:
        x = tf.keras.layers.Rescaling(1. / 255)(input_)
    else:
        x = input_

    # Set the feature extractor layer
    x = feature_extractor(x, training=pretrain_trainable)

    x = tf.keras.layers.Flatten(name='bottleneck')(x)

    # Midle
    dim_reducer = 1
    for i in range(1, n_layers + 1, 1):
        if use_dropout:
            x = tf.keras.layers.Dropout(dropout / i)(x)
            if verbose: print('+ Dropout_' + str(i), 'drop:', str(dropout / i))
        x = tf.keras.layers.Dense(layer_dim // dim_reducer, name='dense_count_' + str(i))(x)
        if verbose: print('+ Dense_' + str(i), 'dim:', str(layer_dim // dim_reducer))
        if use_norm:
            x = tf.keras.layers.BatchNormalization()(x)
            if verbose: print('+ BatchNorm')
        x = tf.keras.layers.Activation(activation)(x)
        dim_reducer = i * layer_reducer
        if verbose: print('\n')

    if output_shape == 1:
        # Output singolo
        f_act = 'sigmoid'
        used_loss = 'binary_crossentropy'
    elif multi_label:
        # Output multi label
        f_act = 'sigmoid'
        used_loss = 'binary_crossentropy'
    else:
        # Output multi classe
        f_act = 'softmax'
        used_loss = 'categorical_crossentropy'

    x = tf.keras.layers.Dropout(dropout / i)(x)
    output_ = tf.keras.layers.Dense(output_shape, activation=f_act)(x)
    if verbose:
        print('+ Final layer: drop', str(dropout / i), ', dim:', str(output_shape))
        print('\nActivation:', f_act, ',Loss', used_loss)

    # Create the new model object
    model = tf.keras.Model(input_, output_, name='DrawClassifier')

    model.compile(optimizer=opt,
                  loss=used_loss,
                  metrics=metrics)
    return model



class Model:
    def __init__(self,folder,file, verbose= False):

        self.clf_NN = define_nn_clf(    pretrain_mod='vgg',
                                       pretrain_trainable=False,
                                       n_layers=2,  # 5,
                                       layer_dim=4096,  # 1024,     #32,
                                       layer_reducer=1,  # 4,
                                       use_dropout=True,
                                       dropout=0.7,
                                       use_norm=False,
                                       activation='relu',  # 'tanh',
                                       multi_label=True,
                                       opt=SGD(learning_rate=1e-5),  # opt, #'rmsprop', #opt,
                                       output_shape=4,
                                       metrics=['accuracy'],
                                       verbose=verbose)

        # Load weights
        file_path = os.path.join(os.getcwd(), folder, file)
        self.clf_NN.load_weights(file_path)
        self.verbose=verbose

        # Load thresholds
        try:
            print('\nLoading threshold file...')
            with open(os.path.join(MODELS_FOLDER, THRESHOLD_FILE), 'rb') as v_file:
                self.thresholds = pickle.load(v_file)
                print('Thresholds', self.thresholds)
            print('Done.\n')

        except:
            print('File', os.path.join(MODELS_FOLDER, THRESHOLD_FILE), 'not found.')
            self.thresholds = [{'label': 'Plan',
                                   'thr': 0.5},
                                  {'label': 'Elevation',
                                   'thr': 0.5},
                                  {'label': 'Section',
                                   'thr': 0.5},
                                  {'label': 'Others',
                                   'thr': 0.5}]


        # Summary
        if verbose:
            self.clf_NN.summary()

    def predict_old(self, image_list, threshold=0.4):
        y_pred = self.clf_NN.predict(np.stack(image_list))

        y_list = []
        for a in y_pred:
            if np.any(a > threshold) == False:
                # Tutti i valori nonsuperano la soglia => cerco il maggiore
                idx = np.argmax(a, axis=-1)
                a = np.zeros(a.shape)
                a[idx] = 1
            else:
                # Almeno un valore supera la soglia
                a = np.where(np.array(a) > threshold, 1, 0).astype('int32')
            y_list.append(a)

        y_pred = np.vstack(y_list).astype(int)

        if self.verbose:
            for idx, pred in enumerate(y_pred):
                print(idx, pred, list(np.array(LABELS)[pred == 1]))

        return [list(np.array(LABELS)[pred == 1]) for pred in y_pred]

    def predict(self, image_list,):
        '''
        Predict applying thresholds.
        If [0,0,0,0] is predicted than the class is the most significative also if it doesn't pass threshold.

        model,
        x_test,
        thresholds

        Return Predicted output (Array of boolean lists), Sure level (boolena: 1 sure, 0 not sure).
        For sure level: prediction are not sure if none of val is grather than thresold => [0,0,0,0] but
        the argmax is taken (level=0, not sure).  If at leas one of the threshold is overhead, than model is sure (level=1)
        '''
        print('Using optimized thresholds')
        y_pred = self.clf_NN.predict(np.stack(image_list))

        level = np.ones(y_pred.shape[0])  # livello di sicurezza della predizione

        # Filter by thresholds to binary output
        thr_arr = np.array([t['thr'] for t in self.thresholds])
        thr_arr = np.tile(thr_arr, (y_pred.shape[0], 1))
        res = y_pred > thr_arr
        res = res.astype('int32')
        # Set sure level (if [0,0,0,0] than not a sure prediction)
        level[np.sum(res, axis=1) == 0] = 0
        # if all output are all zeros than take the one with highter probab.
        res[np.sum(res, axis=1) == 0, np.argmax(y_pred[np.sum(res, axis=1) == 0], axis=1)] = 1

        return [list(np.array(LABELS)[pred == 1]) for pred in res] , list(level)

if __name__=='__main__':
    file_path = os.path.join(os.getcwd(),DATA_FOLDER)
    image_list = [load_and_resize(item, False, False, img_width, img_height, verbose= VERBOSE) for i in
                  [glob.glob(file_path + '/*.%s' % ext) for ext in ["jpg", "gif", "png"]] for item in i]

    # Remove None
    image_list = [x for x in image_list if x is not None]

    print('Loaded', len(image_list), 'images.')
    model=Model(MODELS_FOLDER, MODEL_FILE, verbose=VERBOSE)
    res, level =model.predict(image_list)