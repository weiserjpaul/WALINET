from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv1D, Conv2D, PReLU, Dropout, Input, Dropout, concatenate, MaxPooling2D, Activation, ReLU, MaxPooling2D
from keras.layers import LeakyReLU, Softmax, Cropping2D, UpSampling2D#,regularizers
from keras import initializers as initializers
from keras import regularizers, backend
import tensorflow as tf
#Define the modulable model to train.

#modelCNNseb : model for estimation of either
#              - 13 metabolites (or group of) in spectra 
#              - or simulation parameters  
# at line approx 170
#modelCNNError : estimation of errors on previous estimations
# at line approx 320
#modelCNN_UNet_BL: estimation of baseline of spectra with UNet . input psectrum only

def modelCNNseb(input_shape,output_shape,tLayer, regularizer, nFilters, nNeuron, nMLPlayer, drop) :           #SEB Model based on work from Peter anf Jonhatan
    X_input = Input(input_shape,)
    
    X = Conv2D(filters = 2*nFilters, kernel_size = (11,2), strides = 1, padding ='valid',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X_input)
                     
    #X = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(X)
    #X = Conv2D(filters = nFilters, kernel_size = (104,2), strides = 1, padding ='same',
    #                 data_format='channels_last',
    #                 kernel_initializer= initializers.glorot_normal(seed=0), 
    #                 bias_initializer='zeros')(X_input)

    #X = Conv1D(filters = nFilters,kernel_size= 104, strides=1,padding='same')(X_input)

    if tLayer == "PReLU":
        X = PReLU(shared_axes=[1, 2])(X)
    elif tLayer == "ReLU":
        X = ReLU(shared_axes=[1, 2])(X)
    elif tLayer == "leakyrelu":
        X = LeakyReLU()(X)
    elif tLayer == "sigmoid":
        X = Activation("sigmoid")(X)
    elif tLayer == "tanh":
        X = Activation("tanh")(X)
# layer modified following the np.pad (wrap) of input data
    X = Conv2D(filters = 2*nFilters, kernel_size = (13,2), strides = 1, padding ='valid',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X)
    #X = MaxPooling2D(pool_size=(2,1), strides=None, padding='valid', data_format=None)(X)
    
    if tLayer == "PReLU":
        X = PReLU(shared_axes=[1, 2])(X)
    elif tLayer == "ReLU":
        X = ReLU(shared_axes=[1, 2])(X)
    elif tLayer == "leakyrelu":
        X = LeakyReLU()(X)
    elif tLayer == "sigmoid":
        X = Activation("sigmoid")(X)
    elif tLayer == "tanh":
        X = Activation("tanh")(X)
        
    #may test: change on stride from 2 to 1
    X = Conv2D(filters = int(2.5*nFilters), kernel_size = (13,1), strides = 1, padding ='same',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X)
    X = MaxPooling2D(pool_size=(2,1), strides=None, padding='valid', data_format=None)(X)
    
    
    if tLayer == "PReLU":
        X = PReLU(shared_axes=[1, 2])(X)
    elif tLayer == "ReLU":
        X = ReLU(shared_axes=[1, 2])(X)
    elif tLayer == "leakyrelu":
        X = LeakyReLU()(X)
    elif tLayer == "sigmoid":
        X = Activation("sigmoid")(X)
    elif tLayer == "tanh":
        X = Activation("tanh")(X)
    
    X = Conv2D(filters = int(2.5*nFilters), kernel_size = (13,1), strides = 1, padding ='same',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X)
    X = MaxPooling2D(pool_size=(2,1), strides=None, padding='valid', data_format=None)(X)
    
    if tLayer == "PReLU":
        X = PReLU(shared_axes=[1, 2])(X)
    elif tLayer == "ReLU":
        X = ReLU(shared_axes=[1, 2])(X)
    elif tLayer == "leakyrelu":
        X = LeakyReLU()(X)
    elif tLayer == "sigmoid":
        X = Activation("sigmoid")(X)
    elif tLayer == "tanh":
        X = Activation("tanh")(X)
        
    X = Conv2D(filters = 3*nFilters, kernel_size = (7,1), strides = 1, padding ='valid',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X)                
    #X = MaxPooling2D(pool_size=(2,1), strides=None, padding='valid', data_format=None)(X)

    X = PReLU()(X)
    #added one conv layer valid kernel (9,1), removed last maxpool
    X = Conv2D(filters = 4*nFilters, kernel_size = (9,1), strides = 1, padding ='valid',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X)                
    #X = MaxPooling2D(pool_size=(2,1), strides=None, padding='valid', data_format=None)(X)

    X = PReLU()(X)
                    
#     X = Conv2D(filters = 6*nFilters, kernel_size = (7,1), strides = 1, padding ='valid',
#                      data_format='channels_last',
#                      kernel_initializer= initializers.he_normal(seed=None), 
#                      bias_initializer='zeros')(X)
#     X = PReLU()(X)
    #                 data_format='channels_last',
    #                 kernel_initializer= initializers.he_normal(seed=None), 
    #                 bias_initializer='zeros')(X)
    #X = Conv2D(filters = nFilters, kernel_size = (11,2), strides = 1, padding ='same',
    #                 data_format='channels_last',
    #                 kernel_initializer= initializers.glorot_normal(seed=0), 
    #                 bias_initializer='zeros')(X)
    #X = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(X)
    #X = Softmax(axis=1)(X)
    X = Flatten()(X)
    X = Dropout(drop)(X)
    
    
    X = Dense(int(nNeuron*1.5),kernel_initializer= initializers.glorot_normal(seed=0),  bias_initializer='zeros')(X)
    X = PReLU()(X)    
    
    X = Dense(nNeuron,kernel_initializer= initializers.glorot_normal(seed=0),  bias_initializer='zeros')(X)
    X = PReLU()(X)   
    
    X = Dense(int(nNeuron/2),kernel_initializer= initializers.glorot_normal(seed=0),  bias_initializer='zeros')(X)
    X = PReLU()(X)
        
        #X = Dropout(drop)(X)
    
    # for i in range(nMLPlayer):
#         #if regularizer == 'l2':
#         #    X = Dense(nNeuron,kernel_initializer= initializers.glorot_normal(seed=0),  bias_initializer='zeros',
#         #                kernel_regularizer=regularizers.l2(0.001) )(X)
#         #elif regularizer == 'None':
#         X = Dense(nNeuron,kernel_initializer= initializers.glorot_normal(seed=0),  bias_initializer='zeros')(X)
# 
#         if tLayer == "PReLU":
#             X = PReLU()(X)
#         elif tLayer == "ReLU":
#             X = ReLU()(X)
#         elif tLayer == "leakylayer":
#             X = LeakyReLU()(X)
#         elif tLayer == "sigmoid":
#             X = Activation("sigmoid")(X)
#         elif tLayer == "tanh":
#             X = Activation("tanh")(X)
#         #X = Dropout(drop)(X)

    if regularizer == 'l2':
        Y = Dense(output_shape, kernel_initializer= initializers.glorot_normal(seed=0),  bias_initializer='zeros')(X)
    else:
        Y = Dense(output_shape, kernel_initializer= initializers.glorot_normal(seed=0),  bias_initializer='zeros',
        kernel_regularizer=regularizers.l2(0.01))(X)

    # Create model
    model = Model(inputs = X_input, outputs = Y, name='CNN_Seb')
    return model 

###### based on previous CNN but more filters to cope with increased number of parameters to estimate
def modelCNN_Mets_Params(input_shape,output_shape,tLayer, regularizer, nFilters, nNeuron, nMLPlayer, drop) :           
    X_input = Input(input_shape,)
    
    X = Conv2D(filters = 3*nFilters, kernel_size = (11,2), strides = 1, padding ='valid',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X_input)               

    if tLayer == "PReLU":
        X = PReLU(shared_axes=[1, 2])(X)
    elif tLayer == "ReLU":
        X = ReLU(shared_axes=[1, 2])(X)
 
    X = Conv2D(filters = 3*nFilters, kernel_size = (13,2), strides = 1, padding ='valid',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X)
    
    if tLayer == "PReLU":
        X = PReLU(shared_axes=[1, 2])(X)
    elif tLayer == "ReLU":
        X = ReLU(shared_axes=[1, 2])(X)
            
    X = Conv2D(filters = int(4*nFilters), kernel_size = (13,1), strides = 1, padding ='same',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X)
    X = MaxPooling2D(pool_size=(2,1), strides=None, padding='valid', data_format=None)(X)
        
    if tLayer == "PReLU":
        X = PReLU(shared_axes=[1, 2])(X)
    elif tLayer == "ReLU":
        X = ReLU(shared_axes=[1, 2])(X)
    
    X = Conv2D(filters = int(4*nFilters), kernel_size = (13,1), strides = 1, padding ='same',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X)
    X = MaxPooling2D(pool_size=(2,1), strides=None, padding='valid', data_format=None)(X)
    
    if tLayer == "PReLU":
        X = PReLU(shared_axes=[1, 2])(X)
    elif tLayer == "ReLU":
        X = ReLU(shared_axes=[1, 2])(X)
        
    X = Conv2D(filters = 5*nFilters, kernel_size = (7,1), strides = 1, padding ='valid',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X)                

    X = PReLU()(X)

    X = Conv2D(filters = 5*nFilters, kernel_size = (9,1), strides = 1, padding ='valid',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X)                

    X = PReLU()(X)
                    
    X = Flatten()(X)
    X = Dropout(drop)(X)    
    
    X = Dense(int(nNeuron*1.5),kernel_initializer= initializers.glorot_normal(seed=0),  bias_initializer='zeros')(X)
    X = PReLU()(X)    
    
    X = Dense(nNeuron,kernel_initializer= initializers.glorot_normal(seed=0),  bias_initializer='zeros')(X)
    X = PReLU()(X)   
    
    X = Dense(int(nNeuron/2),kernel_initializer= initializers.glorot_normal(seed=0),  bias_initializer='zeros')(X)
    X = PReLU()(X)

    if regularizer == 'l2':
        Y = Dense(output_shape, kernel_initializer= initializers.glorot_normal(seed=0),  bias_initializer='zeros')(X)
    else:
        Y = Dense(output_shape, kernel_initializer= initializers.glorot_normal(seed=0),  bias_initializer='zeros',
        kernel_regularizer=regularizers.l2(0.01))(X)

    # Create model
    model = Model(inputs = X_input, outputs = Y, name='CNN_Mets_Params')
    return model 
###################test model with large kernels also for encoding, two pathways ##################
def modelCNN_Mets_Params_twopathways(input_shape,output_shape,tLayer, regularizer, nFilters, nNeuron, nMLPlayer, drop) :           #SEB Model based on work from Peter anf Jonhatan
    X_input = Input(input_shape,)
    
    XlargeF = Conv2D(filters = 3*nFilters, kernel_size = (114,2), strides = 1, padding ='valid',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X_input)               
    XlargeF = PReLU()(XlargeF)
    XlargeF = Conv2D(filters = 3*nFilters, kernel_size = (1,2), strides = 1, padding ='valid',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(XlargeF)               
    XlargeF = PReLU()(XlargeF)

    XlargeF = Flatten()(XlargeF) 

    X = Conv2D(filters = 3*nFilters, kernel_size = (11,2), strides = 1, padding ='valid',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X_input)               

    if tLayer == "PReLU":
        X = PReLU(shared_axes=[1, 2])(X)
    elif tLayer == "ReLU":
        X = ReLU(shared_axes=[1, 2])(X)
 
    X = Conv2D(filters = 3*nFilters, kernel_size = (13,2), strides = 1, padding ='valid',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X)
    
    if tLayer == "PReLU":
        X = PReLU(shared_axes=[1, 2])(X)
    elif tLayer == "ReLU":
        X = ReLU(shared_axes=[1, 2])(X)
            
    X = Conv2D(filters = int(4*nFilters), kernel_size = (13,1), strides = 1, padding ='same',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X)
    X = MaxPooling2D(pool_size=(2,1), strides=None, padding='valid', data_format=None)(X)
        
    if tLayer == "PReLU":
        X = PReLU(shared_axes=[1, 2])(X)
    elif tLayer == "ReLU":
        X = ReLU(shared_axes=[1, 2])(X)
    
    X = Conv2D(filters = int(4*nFilters), kernel_size = (13,1), strides = 1, padding ='same',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X)
    X = MaxPooling2D(pool_size=(2,1), strides=None, padding='valid', data_format=None)(X)
    
    if tLayer == "PReLU":
        X = PReLU(shared_axes=[1, 2])(X)
    elif tLayer == "ReLU":
        X = ReLU(shared_axes=[1, 2])(X)
        
    X = Conv2D(filters = 5*nFilters, kernel_size = (7,1), strides = 1, padding ='valid',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X)                

    X = PReLU()(X)

    X = Conv2D(filters = 5*nFilters, kernel_size = (9,1), strides = 1, padding ='valid',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X)                

    X = PReLU()(X)
                    
    X = Flatten()(X)
    
    X = concatenate([X,XlargeF],axis=1)
    X = Dropout(drop)(X)    
    


    X = Dense(int(nNeuron*1.5),kernel_initializer= initializers.glorot_normal(seed=0),  bias_initializer='zeros')(X)
    X = PReLU()(X)    
    
    X = Dense(nNeuron,kernel_initializer= initializers.glorot_normal(seed=0),  bias_initializer='zeros')(X)
    X = PReLU()(X)   
    
    X = Dense(int(nNeuron/2),kernel_initializer= initializers.glorot_normal(seed=0),  bias_initializer='zeros')(X)
    X = PReLU()(X)

    if regularizer == 'l2':
        Y = Dense(output_shape, kernel_initializer= initializers.glorot_normal(seed=0),  bias_initializer='zeros')(X)
    else:
        Y = Dense(output_shape, kernel_initializer= initializers.glorot_normal(seed=0),  bias_initializer='zeros',
        kernel_regularizer=regularizers.l2(0.01))(X)

    # Create model
    model = Model(inputs = X_input, outputs = Y, name='CNN_Mets_Params')
    return model     
###################test about no shared axes on PReLU #################################
def modelCNN_Mets_Params_nosharedax(input_shape,output_shape,tLayer, regularizer, nFilters, nNeuron, nMLPlayer, drop) :           #SEB Model based on work from Peter anf Jonhatan
    X_input = Input(input_shape,)
    
    X = Conv2D(filters = 3*nFilters, kernel_size = (11,2), strides = 1, padding ='valid',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X_input)               

    if tLayer == "PReLU":
        X = PReLU()(X)
        #X = PReLU(shared_axes=[1, 2])(X)
    elif tLayer == "ReLU":
        X = ReLU(shared_axes=[1, 2])(X)
 
    X = Conv2D(filters = 3*nFilters, kernel_size = (13,2), strides = 1, padding ='valid',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X)
    
    if tLayer == "PReLU":
        X = PReLU()(X)
        #X = PReLU(shared_axes=[1, 2])(X)
    elif tLayer == "ReLU":
        X = ReLU(shared_axes=[1, 2])(X)
            
    X = Conv2D(filters = int(4*nFilters), kernel_size = (13,1), strides = 1, padding ='same',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X)
    X = MaxPooling2D(pool_size=(2,1), strides=None, padding='valid', data_format=None)(X)
        
    if tLayer == "PReLU":
        X = PReLU()(X)
        #X = PReLU(shared_axes=[1, 2])(X)
    elif tLayer == "ReLU":
        X = ReLU(shared_axes=[1, 2])(X)
    
    X = Conv2D(filters = int(4*nFilters), kernel_size = (13,1), strides = 1, padding ='same',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X)
    X = MaxPooling2D(pool_size=(2,1), strides=None, padding='valid', data_format=None)(X)
    
    if tLayer == "PReLU":
        X = PReLU()(X)
        #X = PReLU(shared_axes=[1, 2])(X)
    elif tLayer == "ReLU":
        X = ReLU(shared_axes=[1, 2])(X)
        
    X = Conv2D(filters = 5*nFilters, kernel_size = (7,1), strides = 1, padding ='valid',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X)                

    X = PReLU()(X)

    X = Conv2D(filters = 5*nFilters, kernel_size = (9,1), strides = 1, padding ='valid',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X)                

    X = PReLU()(X)
                    
    X = Flatten()(X)
    X = Dropout(drop)(X)    
    
    X = Dense(int(nNeuron*1.5),kernel_initializer= initializers.glorot_normal(seed=0),  bias_initializer='zeros')(X)
    X = PReLU()(X)    
    
    X = Dense(nNeuron,kernel_initializer= initializers.glorot_normal(seed=0),  bias_initializer='zeros')(X)
    X = PReLU()(X)   
    
    X = Dense(int(nNeuron/2),kernel_initializer= initializers.glorot_normal(seed=0),  bias_initializer='zeros')(X)
    X = PReLU()(X)

    if regularizer == 'l2':
        Y = Dense(output_shape, kernel_initializer= initializers.glorot_normal(seed=0),  bias_initializer='zeros')(X)
    else:
        Y = Dense(output_shape, kernel_initializer= initializers.glorot_normal(seed=0),  bias_initializer='zeros',
        kernel_regularizer=regularizers.l2(0.01))(X)

    # Create model
    model = Model(inputs = X_input, outputs = Y, name='CNN_Mets_Params')
    return model 
####################################################
####################################################


def modelCNNError(input1_shape,input2_shape,output_shape,tLayer, regularizer, nFilters, nNeuron, nMLPlayer, drop) :           #SEB Model based on work from Peter anf Jonhatan
    X1_input = Input(input1_shape,)
    X2_input = Input(input2_shape,)


    X = Conv2D(filters = 2*nFilters, kernel_size = (11,2), strides = 1, padding ='valid',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X1_input)
                     
    #X = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(X)
    #X = Conv2D(filters = nFilters, kernel_size = (104,2), strides = 1, padding ='same',
    #                 data_format='channels_last',
    #                 kernel_initializer= initializers.glorot_normal(seed=0), 
    #                 bias_initializer='zeros')(X_input)

    #X = Conv1D(filters = nFilters,kernel_size= 104, strides=1,padding='same')(X_input)

    if tLayer == "PReLU":
        X = PReLU(shared_axes=[1, 2])(X)
    elif tLayer == "ReLU":
        X = ReLU(shared_axes=[1, 2])(X)
    elif tLayer == "leakyrelu":
        X = LeakyReLU()(X)
    elif tLayer == "sigmoid":
        X = Activation("sigmoid")(X)
    elif tLayer == "tanh":
        X = Activation("tanh")(X)
# layer modified following the np.pad (wrap) of input data
    X = Conv2D(filters = 2*nFilters, kernel_size = (13,2), strides = 1, padding ='valid',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X)
    #X = MaxPooling2D(pool_size=(2,1), strides=None, padding='valid', data_format=None)(X)
    
    if tLayer == "PReLU":
        X = PReLU(shared_axes=[1, 2])(X)
    elif tLayer == "ReLU":
        X = ReLU(shared_axes=[1, 2])(X)
    elif tLayer == "leakyrelu":
        X = LeakyReLU()(X)
    elif tLayer == "sigmoid":
        X = Activation("sigmoid")(X)
    elif tLayer == "tanh":
        X = Activation("tanh")(X)
        
    #may test: change on stride from 2 to 1
    X = Conv2D(filters = int(2.5*nFilters), kernel_size = (13,1), strides = 1, padding ='same',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X)
    X = MaxPooling2D(pool_size=(2,1), strides=None, padding='valid', data_format=None)(X)
    
    
    if tLayer == "PReLU":
        X = PReLU(shared_axes=[1, 2])(X)
    elif tLayer == "ReLU":
        X = ReLU(shared_axes=[1, 2])(X)
    elif tLayer == "leakyrelu":
        X = LeakyReLU()(X)
    elif tLayer == "sigmoid":
        X = Activation("sigmoid")(X)
    elif tLayer == "tanh":
        X = Activation("tanh")(X)
    
    X = Conv2D(filters = int(2.5*nFilters), kernel_size = (13,1), strides = 1, padding ='same',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X)
    X = MaxPooling2D(pool_size=(2,1), strides=None, padding='valid', data_format=None)(X)
    
    if tLayer == "PReLU":
        X = PReLU(shared_axes=[1, 2])(X)
    elif tLayer == "ReLU":
        X = ReLU(shared_axes=[1, 2])(X)
    elif tLayer == "leakyrelu":
        X = LeakyReLU()(X)
    elif tLayer == "sigmoid":
        X = Activation("sigmoid")(X)
    elif tLayer == "tanh":
        X = Activation("tanh")(X)
        
    X = Conv2D(filters = 3*nFilters, kernel_size = (7,1), strides = 1, padding ='valid',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X)                
    X = MaxPooling2D(pool_size=(2,1), strides=None, padding='valid', data_format=None)(X)

    X = PReLU()(X)
                    
#     X = Conv2D(filters = 6*nFilters, kernel_size = (7,1), strides = 1, padding ='valid',
#                      data_format='channels_last',
#                      kernel_initializer= initializers.he_normal(seed=None), 
#                      bias_initializer='zeros')(X)
#     X = PReLU()(X)
    #                 data_format='channels_last',
    #                 kernel_initializer= initializers.he_normal(seed=None), 
    #                 bias_initializer='zeros')(X)
    #X = Conv2D(filters = nFilters, kernel_size = (11,2), strides = 1, padding ='same',
    #                 data_format='channels_last',
    #                 kernel_initializer= initializers.glorot_normal(seed=0), 
    #                 bias_initializer='zeros')(X)
    #X = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(X)
    #X = Softmax(axis=1)(X)
    X = Flatten()(X)
    X = Dropout(drop)(X)
    #print("X2 input: ", X2_input.shape)
    #X2_input_flat = Flatten()(X2_input)
    #X = concatenate([X,X2_input_flat],axis=1)
    X2_input_flat = Flatten()(X2_input)
    X = concatenate([X,X2_input_flat],axis=1)

    X = Dense(int(nNeuron*1.5),kernel_initializer= initializers.glorot_normal(seed=None),  bias_initializer='zeros')(X)
    X = PReLU()(X)    
    
    X = concatenate([X,X2_input_flat],axis=1)

    X = Dense(int(nNeuron*1.5),kernel_initializer= initializers.glorot_normal(seed=None),  bias_initializer='zeros')(X)
    X = PReLU()(X)    
    
    X = concatenate([X,X2_input_flat],axis=1)
    
    X = Dense(int(nNeuron*1.5),kernel_initializer= initializers.glorot_normal(seed=None),  bias_initializer='zeros')(X)
    X = PReLU()(X) 

    X = concatenate([X,X2_input_flat],axis=1)
    
    X = Dense(nNeuron,kernel_initializer= initializers.glorot_normal(seed=None), bias_initializer='zeros')(X)
    X = PReLU()(X)   

    X = concatenate([X,X2_input_flat],axis=1)
    
    X = Dense(int(nNeuron/2),kernel_initializer= initializers.glorot_normal(seed=None),  bias_initializer='zeros')(X)
    X = PReLU()(X)
        
        #X = Dropout(drop)(X)
    
    # for i in range(nMLPlayer):
#         #if regularizer == 'l2':
#         #    X = Dense(nNeuron,kernel_initializer= initializers.glorot_normal(seed=0),  bias_initializer='zeros',
#         #                kernel_regularizer=regularizers.l2(0.001) )(X)
#         #elif regularizer == 'None':
#         X = Dense(nNeuron,kernel_initializer= initializers.glorot_normal(seed=0),  bias_initializer='zeros')(X)
# 
#         if tLayer == "PReLU":
#             X = PReLU()(X)
#         elif tLayer == "ReLU":
#             X = ReLU()(X)
#         elif tLayer == "leakylayer":
#             X = LeakyReLU()(X)
#         elif tLayer == "sigmoid":
#             X = Activation("sigmoid")(X)
#         elif tLayer == "tanh":
#             X = Activation("tanh")(X)
#         #X = Dropout(drop)(X)

    if regularizer == 'l2':
        Y = Dense(output_shape, kernel_initializer= initializers.glorot_normal(seed=0),  bias_initializer='zeros')(X)
    else:
        Y = Dense(output_shape, kernel_initializer= initializers.glorot_normal(seed=0),  bias_initializer='zeros',
        kernel_regularizer=regularizers.l2(0.01))(X)

    # Create model
    model = Model(inputs = [X1_input, X2_input], outputs = Y, name='CNN_Error_estimation')
    return model 

###################################
#################### estimation of baseline of spectra (UNet)
###################################  
def modelCNN_UNet_BL(input_shape,tLayer, regularizer, nFilters, nNeuron, nMLPlayer, drop) :          
    #input mx104x3 (real imag real) 
    ##### downscaling to features
    X_input = Input(input_shape,)
    
    l1Conv1 = Conv2D(filters = 2*nFilters, kernel_size = (11,2), strides = 1, padding ='same',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X_input)
    
    l1Conv1Act = PReLU(shared_axes=[1, 2]
                     )(l1Conv1)
    
# layer modified following the np.pad (wrap) of input data
    l1Conv2 = Conv2D(filters = 2*nFilters, kernel_size = (11,2), strides = 1, padding ='same',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(l1Conv1Act)
    #X = MaxPooling2D(pool_size=(2,1), strides=None, padding='valid', data_format=None)(X)
    l1Conv2Act = PReLU(shared_axes=[1, 2]
                     )(l1Conv2)
    
    l1Maxpool = MaxPooling2D(pool_size=(2,1), strides=None, padding='same', data_format=None)(l1Conv2Act)
    # maxpool2 size m,52,1
    
    #may test: change on stride from 2 to 1
    l2Conv1 = Conv2D(filters = int(2.5*nFilters), kernel_size = (7,1), strides = 1, padding ='same',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(l1Maxpool)
    l2Conv1Act = PReLU(shared_axes=[1, 2])(l2Conv1)
    
    l2Conv2 = Conv2D(filters = int(2.5*nFilters), kernel_size = (7,1), strides = 1, padding ='same',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(l2Conv1Act)
    l2Conv2Act = PReLU(shared_axes=[1, 2])(l2Conv2)    
    
    l2Maxpool = MaxPooling2D(pool_size=(2,1), strides=None, padding='same', data_format=None)(l2Conv2Act)
    
    l3Conv1 = Conv2D(filters = 3*nFilters, kernel_size = (7,1), strides = 1, padding ='same',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(l2Maxpool)   
    l3Conv1Act = PReLU(shared_axes=[1, 2])(l3Conv1)
    
    l3Conv2 = Conv2D(filters = 3*nFilters, kernel_size = (7,1), strides = 1, padding ='same',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(l3Conv1Act)   
    l3Conv2Act = PReLU(shared_axes=[1, 2])(l3Conv2)    
                         
    l3Maxpool = MaxPooling2D(pool_size=(2,1), strides=None, padding='same', data_format=None)(l3Conv2Act)
    
    l4Conv1 = Conv2D(filters = 3*nFilters, kernel_size = (7,2), strides = 1, padding ='same',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(l3Maxpool)   
    l4Conv1Act = PReLU(shared_axes=[1, 2])(l4Conv1)
    
    l4Conv2 = Conv2D(filters = 3*nFilters, kernel_size = (13,2), strides = 1, padding ='same',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(l4Conv1Act)   
    l4Conv2Act = PReLU(#shared_axes=[1, 2]
                     )(l4Conv2)    

    ##### upscaling back to image space
    #up6 = Conv2D(int(3*nFilters), kernel_size = (7,1), padding = 'same', kernel_initializer = 'he_normal')(Act5)
    
    l5up = UpSampling2D(size = (2,1))(l4Conv2Act)
    l5merge = concatenate([l3Conv2Act,l5up], axis = 3)
    l5Conv1 = Conv2D(int(2.5*nFilters), kernel_size = (7,2), padding = 'same', kernel_initializer = 'he_normal')(l5merge)
    l5Conv1Act = PReLU(shared_axes=[1, 2])(l5Conv1)
    l5Conv2 = Conv2D(int(2.5*nFilters), kernel_size = (7,2), padding = 'same', kernel_initializer = 'he_normal')(l5Conv1Act)
    l5Conv2Act = PReLU(shared_axes=[1, 2])(l5Conv2)
    
    
    l6up = UpSampling2D(size = (2,1))(l5Conv2Act)
    l6merge = concatenate([l2Conv2Act,l6up], axis = 3)
    l6Conv1 = Conv2D(int(2*nFilters), kernel_size = (7,2), padding = 'same', kernel_initializer = 'he_normal')(l6merge)
    l6Conv1Act = PReLU(shared_axes=[1, 2])(l6Conv1)
    l6Conv2 = Conv2D(int(2*nFilters), kernel_size = (7,2), padding = 'same', kernel_initializer = 'he_normal')(l6Conv1Act)
    l6Conv2Act = PReLU(shared_axes=[1, 2])(l6Conv2)
        
    l7up = UpSampling2D(size = (2,1))(l6Conv2Act)
    l7merge = concatenate([l1Conv2Act,l7up], axis = 3)
    #l7crop = Cropping2D(cropping=((0, 0), (0, 1)))(l7merge)
    l7Conv1 = Conv2D(int(2*nFilters), kernel_size = (7,2), padding = 'same', kernel_initializer = 'he_normal')(l7merge)
    l7Conv1Act = PReLU(shared_axes=[1, 2])(l7Conv1)
    l7Conv2 = Conv2D(int(2*nFilters), kernel_size = (7,2), padding = 'same', kernel_initializer = 'he_normal')(l7Conv1Act)
    l7Conv2Act = PReLU(#shared_axes=[1, 2]
                      )(l7Conv2)  
    l7Conv3 = Conv2D(int(2*nFilters), kernel_size = (1,2), padding = 'valid', kernel_initializer = 'he_normal')(l7Conv2Act)
    l7Conv3Act = PReLU(shared_axes=[1, 2]
                      )(l7Conv3)    
    
    
    convEnd = Conv2D(1, 1, kernel_initializer = 'he_normal')(l7Conv3Act)
    convEndAct = PReLU(shared_axes=[1, 2])(convEnd) 
    #Flat6 = Flatten()(Act5)
    #Drop6 = Dropout(drop)(Flat6)
    
    
    #Dense7 = Dense(int(nNeuron*1.5),kernel_initializer= initializers.glorot_normal(seed=0),  bias_initializer='zeros')(Drop6)
    #Act7 = PReLU()(Drop6)    
    
    #Dense8 = Dense(nNeuron,kernel_initializer= initializers.glorot_normal(seed=0),  bias_initializer='zeros')(Act7)
    #Act8 = PReLU()(Dense8)   
    
    #Dense9 = Dense(int(nNeuron/2),kernel_initializer= initializers.glorot_normal(seed=0),  bias_initializer='zeros')(Act8)
    #Act9 = PReLU()(Dense9)
     
    #if regularizer == 'l2':
    #    Y = Dense(14, kernel_initializer= initializers.glorot_normal(seed=0),  bias_initializer='zeros')(Act9)
    #else:
    #    Y = Dense(14, kernel_initializer= initializers.glorot_normal(seed=0),  bias_initializer='zeros',
    #    kernel_regularizer=regularizers.l2(0.01))(Act9)
    
    print("pass model")
    # Create model
    model = Model(inputs = X_input, outputs = convEndAct, name='CNN_Unet_BL')
    return model 

###################################
#################### correction of Lipid suppression distortion (UNet)
###################################  
def model_UNet_Lipid_sup_dist_Correction(inputLip_shape,inputCorr_shape,tLayer, regularizer, nFilters, nNeuron, nMLPlayer, drop) :          
    #2x inputs of (examples x 104 x 3) :  3 = (real imag real) 
    X1_input = Input(inputLip_shape,)
    X2_input = Input(inputCorr_shape,)
    
    ##### encoding to features of lipid contaminated spectra    
    ## 1st stage
    X1s1 = Conv2D(filters = 2*nFilters, kernel_size = (11,2), strides = 1, padding ='same',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X1_input)
    X1s1 = PReLU(shared_axes=[1, 2]
                     )(X1s1)
    
    X1s1 = Conv2D(filters = 2*nFilters, kernel_size = (11,2), strides = 1, padding ='same',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X1s1)
    X1s1 = PReLU(shared_axes=[1, 2]
                     )(X1s1)

    ## 2nd stage, lipid contaminated spectra 
    X1s2 = MaxPooling2D(pool_size=(2,1), strides=None, padding='same', data_format=None)(X1s1)
    
    X1s2 = Conv2D(filters = int(2.5*nFilters), kernel_size = (7,1), strides = 1, padding ='same',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X1s2)
    X1s2 = PReLU(shared_axes=[1, 2])(X1s2)
    
    X1s2 = Conv2D(filters = int(2.5*nFilters), kernel_size = (7,1), strides = 1, padding ='same',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X1s2)
    X1s2 = PReLU(shared_axes=[1, 2])(X1s2)    

    ## 3th stage, lipid contaminated spectra
    X1s3 = MaxPooling2D(pool_size=(2,1), strides=None, padding='same', data_format=None)(X1s2)
    
    X1s3 = Conv2D(filters = 3*nFilters, kernel_size = (7,1), strides = 1, padding ='same',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X1s3)   
    X1s3 = PReLU(shared_axes=[1, 2])(X1s3)
    
    X1s3 = Conv2D(filters = 3*nFilters, kernel_size = (7,1), strides = 1, padding ='same',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X1s3)   
    X1s3 = PReLU(shared_axes=[1, 2])(X1s3)    

    ## 4th stage, lipid contaminated spectra                         
    X1s4 = MaxPooling2D(pool_size=(2,1), strides=None, padding='same', data_format=None)(X1s3)
    
    X1s4 = Conv2D(filters = 3*nFilters, kernel_size = (7,2), strides = 1, padding ='same',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X1s4)   
    X1s4 = PReLU(shared_axes=[1, 2])(X1s4)
    
    X1s4 = Conv2D(filters = 3*nFilters, kernel_size = (13,2), strides = 1, padding ='same',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X1s4)   
    X1s4 = PReLU(#shared_axes=[1, 2]
                     )(X1s4)    

    ##### encoding to features of corrected spectra 
    ## 1st stage
    X2s1 = Conv2D(filters = 2*nFilters, kernel_size = (11,2), strides = 1, padding ='same',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X2_input)
    X2s1 = PReLU(shared_axes=[1, 2]
                     )(X2s1)
    
    X2s1 = Conv2D(filters = 2*nFilters, kernel_size = (11,2), strides = 1, padding ='same',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X2s1)
    X2s1 = PReLU(shared_axes=[1, 2]
                     )(X2s1)

    ## 2nd stage, corrected spectra
    X2s2 = MaxPooling2D(pool_size=(2,1), strides=None, padding='same', data_format=None)(X2s1)
     
    X2s2 = Conv2D(filters = int(2.5*nFilters), kernel_size = (7,1), strides = 1, padding ='same',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X2s2)
    X2s2 = PReLU(shared_axes=[1, 2])(X2s2)
    
    X2s2 = Conv2D(filters = int(2.5*nFilters), kernel_size = (7,1), strides = 1, padding ='same',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X2s2)
    X2s2 = PReLU(shared_axes=[1, 2])(X2s2)    
    
    ## 3th stage, corrected spectra
    X2s3 = MaxPooling2D(pool_size=(2,1), strides=None, padding='same', data_format=None)(X2s2)
    
    X2s3 = Conv2D(filters = 3*nFilters, kernel_size = (7,1), strides = 1, padding ='same',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X2s3)   
    X2s3 = PReLU(shared_axes=[1, 2])(X2s3)
    
    X2s3 = Conv2D(filters = 3*nFilters, kernel_size = (7,1), strides = 1, padding ='same',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X2s3)   
    X2s3 = PReLU(shared_axes=[1, 2])(X2s3)    

    ## 4th stage, corrected spectra
    X2s4 = MaxPooling2D(pool_size=(2,1), strides=None, padding='same', data_format=None)(X2s3)
    
    X2s4 = Conv2D(filters = 3*nFilters, kernel_size = (7,2), strides = 1, padding ='same',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X2s4)   
    X2s4 = PReLU(shared_axes=[1, 2])(X2s4)
    
    X2s4 = Conv2D(filters = 3*nFilters, kernel_size = (13,2), strides = 1, padding ='same',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X2s4)   
    X2s4 = PReLU(#shared_axes=[1, 2]
                     )(X2s4) 
    
    ##### mixing of encoded layers (from X1 and X2)
    # 4th stage mixing
    X1X2s4 = concatenate([X1s4,X2s4], axis = 2)
    X1X2s4 = Flatten()(X1X2s4)
    X1X2s4 = Dropout(drop)(X1X2s4) 
    X1X2s4 = Dense(tf.keras.backend.int_shape(X1X2s4)[1],kernel_initializer= initializers.glorot_normal(),  bias_initializer='zeros')(X1X2s4)
    X1X2s4 = PReLU()(X1X2s4) 
    X1X2s4 = Dense(tf.keras.backend.int_shape(X1X2s4)[1],kernel_initializer= initializers.glorot_normal(),  bias_initializer='zeros')(X1X2s4)
    X1X2s4 = PReLU()(X1X2s4)   
    X1X2s4 = tf.keras.backend.expand_dims(X1X2s4, axis=-1)
    X1X2s4 = tf.keras.backend.expand_dims(X1X2s4, axis=-1)
    ##### upscaling back to image space
    # 5th stage: upscaling to shape of "stage 3" and concat with corrected spectra (X2) stage 3
    
    s5 = UpSampling2D(size = (2,1))(X1X2s4)
    s5 = concatenate([X2s3,s5], axis = 3)
    s5 = Conv2D(int(2.5*nFilters), kernel_size = (7,2), padding = 'same', kernel_initializer = 'he_normal')(s5)
    s5 = PReLU(shared_axes=[1, 2])(s5)
    s5 = Conv2D(int(2.5*nFilters), kernel_size = (7,2), padding = 'same', kernel_initializer = 'he_normal')(s5)
    s5 = PReLU(shared_axes=[1, 2])(s5)
    
    # 6th stage: upscaling to shape of "stage 2" and concat with corrected spectra (X2) stage 2
    s6 = UpSampling2D(size = (2,1))(s5)
    s6 = concatenate([X2s2, s6], axis = 3)
    s6 = Conv2D(int(2*nFilters), kernel_size = (7,2), padding = 'same', kernel_initializer = 'he_normal')(s6)
    s6 = PReLU(shared_axes=[1, 2])(s6)
    s6 = Conv2D(int(2*nFilters), kernel_size = (7,2), padding = 'same', kernel_initializer = 'he_normal')(s6)
    s6 = PReLU(shared_axes=[1, 2])(s6)

    # 7th stage: upscaling to shape of "stage 1" and concat with corrected spectra (X2) stage 1    
    s7 = UpSampling2D(size = (2,1))(s6)
    s7 = concatenate([X2s1,s7], axis = 3)
    s7 = Conv2D(int(2*nFilters), kernel_size = (7,2), padding = 'same', kernel_initializer = 'he_normal')(s7)
    s7 = PReLU(shared_axes=[1, 2])(s7)
    s7 = Conv2D(int(2*nFilters), kernel_size = (7,2), padding = 'same', kernel_initializer = 'he_normal')(s7)
    s7 = PReLU(#shared_axes=[1, 2]
                      )(s7)  
    s7 = Conv2D(int(2*nFilters), kernel_size = (1,2), padding = 'valid', kernel_initializer = 'he_normal')(s7)
    s7 = PReLU(shared_axes=[1, 2]
                      )(s7)    
                      
    s7 = Conv2D(1, 1, kernel_initializer = 'he_normal')(s7)
    Y = PReLU(shared_axes=[1, 2])(s7) 
    
    print("pass model")
    # Create model
    model = Model(inputs = [X1_input, X2_input], outputs = Y, name='UNet_Lipid_sup_dist_Correction')
    return model 

#####same as above but with less params
def modelCNN_UNet_BL_test(input_shape,tLayer, regularizer, nFilters, nNeuron, nMLPlayer, drop) :           #SEB Model based on work from Peter anf Jonhatan
    #input mx104x3 (real imag real) 
    ##### downscaling to features
    X_input = Input(input_shape,)
    
    l1Conv1 = Conv2D(filters = 2*nFilters, kernel_size = (11,2), strides = 1, padding ='same',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X_input)
    
    l1Conv1Act = PReLU(shared_axes=[1, 2])(l1Conv1)
    
# layer modified following the np.pad (wrap) of input data
    #l1Conv2 = Conv2D(filters = 2*nFilters, kernel_size = (11,2), strides = 1, padding ='same',
    #                 data_format='channels_last',
    #                 kernel_initializer= initializers.he_normal(seed=None), 
    #                 bias_initializer='zeros')(l1Conv1Act)
    ##X = MaxPooling2D(pool_size=(2,1), strides=None, padding='valid', data_format=None)(X)
    #l1Conv2Act = PReLU(shared_axes=[1, 2])(l1Conv2)
    
    l1Maxpool = MaxPooling2D(pool_size=(2,1), strides=None, padding='same', data_format=None)(l1Conv1Act)
    # maxpool2 size m,52,1
    
    #may test: change on stride from 2 to 1
    l2Conv1 = Conv2D(filters = int(2.5*nFilters), kernel_size = (11,2), strides = 1, padding ='same',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(l1Maxpool)
    l2Conv1Act = PReLU(shared_axes=[1, 2])(l2Conv1)
    
#     l2Conv2 = Conv2D(filters = int(2.5*nFilters), kernel_size = (7,1), strides = 1, padding ='same',
#                      data_format='channels_last',
#                      kernel_initializer= initializers.he_normal(seed=None), 
#                      bias_initializer='zeros')(l2Conv1Act)
#     l2Conv2Act = PReLU(shared_axes=[1, 2])(l2Conv2)    
    
    l2Maxpool = MaxPooling2D(pool_size=(2,1), strides=None, padding='same', data_format=None)(l2Conv1Act)
    
    l3Conv1 = Conv2D(filters = 3*nFilters, kernel_size = (7,1), strides = 1, padding ='same',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(l2Maxpool)   
    l3Conv1Act = PReLU(shared_axes=[1, 2])(l3Conv1)
    
#     l3Conv2 = Conv2D(filters = 3*nFilters, kernel_size = (7,1), strides = 1, padding ='same',
#                      data_format='channels_last',
#                      kernel_initializer= initializers.he_normal(seed=None), 
#                      bias_initializer='zeros')(l3Conv1Act)   
#     l3Conv2Act = PReLU(shared_axes=[1, 2])(l3Conv2)    
                         
    l3Maxpool = MaxPooling2D(pool_size=(2,1), strides=None, padding='same', data_format=None)(l3Conv1Act)
    
    l4Conv1 = Conv2D(filters = 3*nFilters, kernel_size = (7,1), strides = 1, padding ='same',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(l3Maxpool)   
    l4Conv1Act = PReLU(shared_axes=[1, 2])(l4Conv1)
    
#     l4Conv2 = Conv2D(filters = 3*nFilters, kernel_size = (7,1), strides = 1, padding ='same',
#                      data_format='channels_last',
#                      kernel_initializer= initializers.he_normal(seed=None), 
#                      bias_initializer='zeros')(l4Conv1Act)   
#     l4Conv2Act = PReLU(shared_axes=[1, 2])(l4Conv2)    

    ##### upscaling back to image space
    #up6 = Conv2D(int(3*nFilters), kernel_size = (7,1), padding = 'same', kernel_initializer = 'he_normal')(Act5)
    
    l5up = UpSampling2D(size = (2,1))(l4Conv1Act)
    l5merge = concatenate([l3Conv1Act,l5up], axis = 3)
    l5Conv1 = Conv2D(int(2.5*nFilters), kernel_size = (7,1), padding = 'same', kernel_initializer = 'he_normal')(l5merge)
    l5Conv1Act = PReLU(shared_axes=[1, 2])(l5Conv1)
    l5Conv2 = Conv2D(int(2.5*nFilters), kernel_size = (7,1), padding = 'same', kernel_initializer = 'he_normal')(l5Conv1Act)
    l5Conv2Act = PReLU(shared_axes=[1, 2])(l5Conv2)
    
    
    l6up = UpSampling2D(size = (2,1))(l5Conv2Act)
    l6merge = concatenate([l2Conv1Act,l6up], axis = 3)
    l6Conv1 = Conv2D(int(2*nFilters), kernel_size = (7,1), padding = 'same', kernel_initializer = 'he_normal')(l6merge)
    l6Conv1Act = PReLU(shared_axes=[1, 2])(l6Conv1)
#     l6Conv2 = Conv2D(int(2*nFilters), kernel_size = (7,1), padding = 'same', kernel_initializer = 'he_normal')(l6Conv1Act)
#     l6Conv2Act = PReLU(shared_axes=[1, 2])(l6Conv2)
        
    l7up = UpSampling2D(size = (2,1))(l6Conv1Act)
    l7merge = concatenate([l1Conv1Act,l7up], axis = 3)
    #l7crop = Cropping2D(cropping=((0, 0), (0, 1)))(l7merge)
    l7Conv1 = Conv2D(int(2*nFilters), kernel_size = (7,1), padding = 'same', kernel_initializer = 'he_normal')(l7merge)
    l7Conv1Act = PReLU(shared_axes=[1, 2])(l7Conv1)
    l7Conv2 = Conv2D(int(2*nFilters), kernel_size = (7,1), padding = 'same', kernel_initializer = 'he_normal')(l7Conv1Act)
    l7Conv2Act = PReLU(shared_axes=[1, 2])(l7Conv2)  
    l7Conv3 = Conv2D(int(2*nFilters), kernel_size = (1,2), padding = 'valid', kernel_initializer = 'he_normal')(l7Conv2Act)
    l7Conv3Act = PReLU(shared_axes=[1, 2])(l7Conv3)    
    
    
    convEnd = Conv2D(1, 1, kernel_initializer = 'he_normal')(l7Conv3Act)
    convEndAct = PReLU(shared_axes=[1, 2])(convEnd) 
    #Flat6 = Flatten()(Act5)
    #Drop6 = Dropout(drop)(Flat6)
    
    
    #Dense7 = Dense(int(nNeuron*1.5),kernel_initializer= initializers.glorot_normal(seed=0),  bias_initializer='zeros')(Drop6)
    #Act7 = PReLU()(Drop6)    
    
    #Dense8 = Dense(nNeuron,kernel_initializer= initializers.glorot_normal(seed=0),  bias_initializer='zeros')(Act7)
    #Act8 = PReLU()(Dense8)   
    
    #Dense9 = Dense(int(nNeuron/2),kernel_initializer= initializers.glorot_normal(seed=0),  bias_initializer='zeros')(Act8)
    #Act9 = PReLU()(Dense9)
     
    #if regularizer == 'l2':
    #    Y = Dense(14, kernel_initializer= initializers.glorot_normal(seed=0),  bias_initializer='zeros')(Act9)
    #else:
    #    Y = Dense(14, kernel_initializer= initializers.glorot_normal(seed=0),  bias_initializer='zeros',
    #    kernel_regularizer=regularizers.l2(0.01))(Act9)
    
    print("pass model")
    # Create model
    model = Model(inputs = X_input, outputs = convEndAct, name='CNN_Unet_BL_test')
    return model 
    
def modelCNNError_v2(input1_shape,input2_shape,input3_shape,output_shape,tLayer, regularizer, nFilters, nNeuron, nMLPlayer, drop) :           #SEB Model based on work from Peter anf Jonhatan
    # load json and create model
  
    
    X1_input = Input(input1_shape,)
    X2_input = Input(input2_shape,)
    X3_input = Input(input3_shape)

    X = Conv2D(filters = 2*nFilters, kernel_size = (11,2), strides = 1, padding ='valid',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X1_input)
                     
    #X = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(X)
    #X = Conv2D(filters = nFilters, kernel_size = (104,2), strides = 1, padding ='same',
    #                 data_format='channels_last',
    #                 kernel_initializer= initializers.glorot_normal(seed=0), 
    #                 bias_initializer='zeros')(X_input)

    #X = Conv1D(filters = nFilters,kernel_size= 104, strides=1,padding='same')(X_input)

    if tLayer == "PReLU":
        X = PReLU(shared_axes=[1, 2])(X)
    elif tLayer == "ReLU":
        X = ReLU(shared_axes=[1, 2])(X)
    elif tLayer == "leakyrelu":
        X = LeakyReLU()(X)
    elif tLayer == "sigmoid":
        X = Activation("sigmoid")(X)
    elif tLayer == "tanh":
        X = Activation("tanh")(X)
# layer modified following the np.pad (wrap) of input data
    X = Conv2D(filters = 2*nFilters, kernel_size = (13,2), strides = 1, padding ='valid',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X)
    #X = MaxPooling2D(pool_size=(2,1), strides=None, padding='valid', data_format=None)(X)
    
    if tLayer == "PReLU":
        X = PReLU(shared_axes=[1, 2])(X)
    elif tLayer == "ReLU":
        X = ReLU(shared_axes=[1, 2])(X)
    elif tLayer == "leakyrelu":
        X = LeakyReLU()(X)
    elif tLayer == "sigmoid":
        X = Activation("sigmoid")(X)
    elif tLayer == "tanh":
        X = Activation("tanh")(X)
        
    #may test: change on stride from 2 to 1
    X = Conv2D(filters = int(2.5*nFilters), kernel_size = (13,1), strides = 1, padding ='same',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X)
    X = MaxPooling2D(pool_size=(2,1), strides=None, padding='valid', data_format=None)(X)
    
    
    if tLayer == "PReLU":
        X = PReLU(shared_axes=[1, 2])(X)
    elif tLayer == "ReLU":
        X = ReLU(shared_axes=[1, 2])(X)
    elif tLayer == "leakyrelu":
        X = LeakyReLU()(X)
    elif tLayer == "sigmoid":
        X = Activation("sigmoid")(X)
    elif tLayer == "tanh":
        X = Activation("tanh")(X)
    
    X = Conv2D(filters = int(2.5*nFilters), kernel_size = (13,1), strides = 1, padding ='same',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X)
    X = MaxPooling2D(pool_size=(2,1), strides=None, padding='valid', data_format=None)(X)
    
    if tLayer == "PReLU":
        X = PReLU(shared_axes=[1, 2])(X)
    elif tLayer == "ReLU":
        X = ReLU(shared_axes=[1, 2])(X)
    elif tLayer == "leakyrelu":
        X = LeakyReLU()(X)
    elif tLayer == "sigmoid":
        X = Activation("sigmoid")(X)
    elif tLayer == "tanh":
        X = Activation("tanh")(X)
        
    X = Conv2D(filters = 3*nFilters, kernel_size = (7,1), strides = 1, padding ='valid',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X)                
    X = MaxPooling2D(pool_size=(2,1), strides=None, padding='valid', data_format=None)(X)

    X = PReLU()(X)
    
    X = concatenate([X,X3_input],axis=1)   
    
    X = Conv2D(filters = 3*nFilters, kernel_size = (3,1), strides = 1, padding ='valid',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X)
    X = PReLU()(X)
    X = Conv2D(filters = 3*nFilters, kernel_size = (3,1), strides = 1, padding ='valid',
                     data_format='channels_last',
                     kernel_initializer= initializers.he_normal(seed=None), 
                     bias_initializer='zeros')(X)    
    X = PReLU()(X)         
#     X = Conv2D(filters = 6*nFilters, kernel_size = (7,1), strides = 1, padding ='valid',
#                      data_format='channels_last',
#                      kernel_initializer= initializers.he_normal(seed=None), 
#                      bias_initializer='zeros')(X)
#     X = PReLU()(X)
    #                 data_format='channels_last',
    #                 kernel_initializer= initializers.he_normal(seed=None), 
    #                 bias_initializer='zeros')(X)
    #X = Conv2D(filters = nFilters, kernel_size = (11,2), strides = 1, padding ='same',
    #                 data_format='channels_last',
    #                 kernel_initializer= initializers.glorot_normal(seed=0), 
    #                 bias_initializer='zeros')(X)
    #X = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(X)
    #X = Softmax(axis=1)(X)
    X = Flatten()(X)
    X = Dropout(drop)(X)
    #print("X2 input: ", X2_input.shape)
    #X2_input_flat = Flatten()(X2_input)
    #X = concatenate([X,X2_input_flat],axis=1)
    X2_input_flat = Flatten()(X2_input)
    X = concatenate([X,X2_input_flat],axis=1)

    X = Dense(int(nNeuron*1.5),kernel_initializer= initializers.glorot_normal(seed=None),  bias_initializer='zeros')(X)
    X = PReLU()(X)    
    
    X = concatenate([X,X2_input_flat],axis=1)

    X = Dense(int(nNeuron*1.5),kernel_initializer= initializers.glorot_normal(seed=None),  bias_initializer='zeros')(X)
    X = PReLU()(X)    
    
    X = concatenate([X,X2_input_flat],axis=1)
    
    X = Dense(int(nNeuron*1.5),kernel_initializer= initializers.glorot_normal(seed=None),  bias_initializer='zeros')(X)
    X = PReLU()(X) 

    X = concatenate([X,X2_input_flat],axis=1)
    
    X = Dense(nNeuron,kernel_initializer= initializers.glorot_normal(seed=None), bias_initializer='zeros')(X)
    X = PReLU()(X)   

    X = concatenate([X,X2_input_flat],axis=1)
    
    X = Dense(int(nNeuron/2),kernel_initializer= initializers.glorot_normal(seed=None),  bias_initializer='zeros')(X)
    X = PReLU()(X)
        
        #X = Dropout(drop)(X)
    
    # for i in range(nMLPlayer):
#         #if regularizer == 'l2':
#         #    X = Dense(nNeuron,kernel_initializer= initializers.glorot_normal(seed=0),  bias_initializer='zeros',
#         #                kernel_regularizer=regularizers.l2(0.001) )(X)
#         #elif regularizer == 'None':
#         X = Dense(nNeuron,kernel_initializer= initializers.glorot_normal(seed=0),  bias_initializer='zeros')(X)
# 
#         if tLayer == "PReLU":
#             X = PReLU()(X)
#         elif tLayer == "ReLU":
#             X = ReLU()(X)
#         elif tLayer == "leakylayer":
#             X = LeakyReLU()(X)
#         elif tLayer == "sigmoid":
#             X = Activation("sigmoid")(X)
#         elif tLayer == "tanh":
#             X = Activation("tanh")(X)
#         #X = Dropout(drop)(X)

    if regularizer == 'l2':
        Y = Dense(output_shape, kernel_initializer= initializers.glorot_normal(seed=0),  bias_initializer='zeros')(X)
    else:
        Y = Dense(output_shape, kernel_initializer= initializers.glorot_normal(seed=0),  bias_initializer='zeros',
        kernel_regularizer=regularizers.l2(0.01))(X)

    # Create model
    model = Model(inputs = [X1_input, X2_input, X3_input], outputs = Y, name='CNN_Error_estimation')
    return model     
 
def modelCNNError_vDense(input_shape,output_shape,tLayer, regularizer, nFilters, nNeuron, nMLPlayer, drop) :           #SEB Model based on work from Peter anf Jonhatan
    X_input = Input(input_shape,)
    
    X = Flatten()(X_input)
    
    #print("X2 input: ", X2_input.shape)
    #X2_input_flat = Flatten()(X2_input)
    #X = concatenate([X,X2_input_flat],axis=1)
    X = Dense(int(nNeuron*3),kernel_initializer= initializers.glorot_normal(seed=None),  bias_initializer='zeros')(X)
    X = PReLU()(X)    
    X = Dropout(drop)(X)
    
    X = Dense(int(nNeuron*3),kernel_initializer= initializers.glorot_normal(seed=None),  bias_initializer='zeros')(X)
    X = PReLU()(X)    
    X = Dropout(drop)(X)

    X = Dense(int(nNeuron*2),kernel_initializer= initializers.glorot_normal(seed=None),  bias_initializer='zeros')(X)
    X = PReLU()(X)    
    X = Dropout(drop)(X)

    X = Dense(int(nNeuron*2),kernel_initializer= initializers.glorot_normal(seed=None),  bias_initializer='zeros')(X)
    X = PReLU()(X)    
    X = Dropout(drop)(X)
    
    X = Dense(int(nNeuron),activation='linear', kernel_initializer= initializers.glorot_normal(seed=None),  bias_initializer='zeros')(X)
    #X = PReLU()(X) 
    #X = Dropout(drop)(X)
    
    if regularizer == 'l2':
        Y = Dense(output_shape, kernel_initializer= initializers.glorot_normal(seed=0),  bias_initializer='zeros')(X)
    else:
        Y = Dense(output_shape, kernel_initializer= initializers.glorot_normal(seed=0),  bias_initializer='zeros',
        kernel_regularizer=regularizers.l2(0.01))(X)

    # Create model
    model = Model(inputs = X_input, outputs = Y, name='CNN_Error_estimation_vDense')
    return model 
