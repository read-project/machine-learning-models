# -*- coding: utf-8 -*-
"""Images_Draw_Classif_Predict.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10gBRL9Wdql3a1FE6HGkTCvjAZRRvm1YI

# Images-Draw Classifier

## Common Setup
"""
import os.path

import numpy as np
import tensorflow as tf
from keras import backend as K

from tensorflow.keras.applications import ResNet50, VGG16


AUTOTUNE = tf.data.AUTOTUNE

img_height = 224
img_width = 224
layers = 3
AUTOTUNE = tf.data.AUTOTUNE
LABELS=['Photo', 'Draw']
batch_size = 32

DATA_FOLDER = 'test'  # This is the image and metadata folder
MODELS_FOLDER = 'models'  # This is where you want to store trained models
MODEL_FILE = 'only_weights'

def decode_img_pred(img):
  # Convert the compressed string to a 3D uint8 tensor
  img = tf.io.decode_jpeg(img, channels=3)
  # Resize the image to the desired size
  return tf.image.resize(img, [img_height, img_width])

def process_path_pred(file_path):
  # Load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img_pred(img)
  return img

#Metrics
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

"""## Def Model to tune
[Doc keras tuner](https://towardsdatascience.com/hyperparameter-tuning-with-kerastuner-and-tensorflow-c4a4d690b31a)
"""

def create_model(pretrain_mod='res',
                 pretrain_trainable= False,
                 units= 512,
                 num_layers= 3,
                 activation= 'relu',
                 dropout=[0.1,0.1],
                 global_pooling= 'max',
                 learning_rate = 1e-3,
                 opt_type= 'Adam'
                 ):
  # Initialize the Pretrained Model
  if pretrain_mod== 'res':
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
  x = tf.keras.layers.Rescaling(1./255) (input_)

  # Set the feature extractor layer
  x = feature_extractor(x , training=pretrain_trainable)

  # Set the pooling layer
  if global_pooling == 'max':
    x = tf.keras.layers.GlobalMaxPooling2D(name= 'feature')(x)
  elif global_pooling == 'avg':
    x = tf.keras.layers.GlobalAveragePooling2D(name= 'feature')(x)
  else:
    #Flat layers
    x = tf.keras.layers.Flatten(name= 'feature')(x)

  # Tune the number of hidden layers and units in each.
  # Number of hidden layers: 1 - 5
  # Number of Units: 64 - 512 with stepsize of 32

  for i in range(1, num_layers):
    u_val =units // i
    x= tf.keras.layers.Dense(units=u_val,
                             activation= activation,
                             trainable= not(pretrain_trainable))(x)
    # Tune dropout layer with values from 0 - 0.3 with stepsize of 0.1.
    x= tf.keras.layers.Dropout(dropout[i-1]) (x)



  # Set the final layer with sigmoid activation function
  output_ = tf.keras.layers.Dense(1, activation='sigmoid', trainable= not(pretrain_trainable))(x)

  # Create the new model object
  model = tf.keras.Model(input_, output_)

  if opt_type =='Adam':
    opt=tf.keras.optimizers.legacy.Adam(learning_rate= learning_rate)   #tf.keras.optimizers.Adam(learning_rate= learning_rate)
  elif opt_type =='SGD':
    opt=tf.keras.optimizers.SGD(learning_rate= learning_rate)

  # Compile it
  model.compile(optimizer= opt,
                loss='binary_crossentropy',
                metrics=['accuracy',f1_m])

  return model


class Model:
    def __init__(self,folder,file, verbose= False):
        # Create a new model instance
        self.model = create_model(
            pretrain_mod='vgg',
            # pretrain_trainable=True,
            units=768,
            activation='relu',
            num_layers=3,
            global_pooling='flt',
            # learning_rate= 1e-07,
            opt_type='Adam',
            dropout=[0.2, 0.2, 0.2, 0.2, 0.2]
        )

        try:
            # Restore the weights
            self.model.load_weights(os.path.join(folder, file))
        except:
            print('Error loading weights from',os.path.join(folder, file))

        # Summary
        if verbose:
            self.clf_NN.summary()


    def predict(self, image_list):
        '''
        Predict applying thresholds.
        If [0,0,0,0] is predicted than the class is the most significative also if it doesn't pass threshold.

        model,
        x_test,
        thresholds

        Return Predicted output (Array of boolean lists), Sure level (boolena: 1 sure, 0 not sure).
        For sure level: prediction are not sure il non of val is grather than thresold => [0,0,0,0] but
        the argmax is taken.
        '''
        print('Using optimized thresholds')
        perc = self.model.predict(np.stack(image_list))
        y_pred = np.round(perc).astype(int)

        return [list(np.array(LABELS)[pred]) for pred in y_pred]

'''
if __name__=='__main__':
    # Create a new model instance
    model = Model (MODELS_FOLDER,MODEL_FILE)

    data_dir=pathlib.Path(DATA_FOLDER)
    print('Found',len(list(data_dir.glob('*.jpg'))),'images.')

    list_ds = tf.data.Dataset.list_files(str(data_dir/'*.jpg'), shuffle=False)

    test_ds = list_ds.map(process_path_pred, num_parallel_calls=AUTOTUNE)

    from time import perf_counter
    fig = plt.figure(figsize=(28, 10))
    columns = 9
    rows = 4

    # ax enables access to manipulate each of subplots
    ax = []

    #Run over all test set and display
    pred_labels=[]
    i=0

    #file_list = [d.numpy() for d in list_ds]
    img_list=[]
    for image in test_ds:
        try:
            print(type(image))
            img_list.append(image)
        except:
            #print ('error converting',file_list[idx])
            print('error on file n.')

    res=model.predict(img_list)


    for image in test_ds:
      perc = model.predict(np.expand_dims(image, axis=0))
      y_pred= np.round(perc).astype(int)[0][0]

      img= image.numpy().astype("uint8")

      ax.append( fig.add_subplot(rows, columns, i+1) )
      ax[-1].set_title(LABELS[y_pred] + ' ' + str(np.round(perc, 3)[0]))  # set title
      ax[-1].axis("off")

      plt.imshow(img)

      pred_labels.append(y_pred)
      i+=1

    plt.show()
'''