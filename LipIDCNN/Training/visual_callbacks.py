from keras.callbacks import Callback
import matplotlib.pyplot as plt    
import matplotlib.patches as mpatches  
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import tools

#Object to display training result/value during training
#Not robuste. Depend a lot of your data/model.

class trainPlotter(Callback):
    """Plot training Accuracy and Loss values on a Matplotlib graph. 

    The graph is updated by the 'on_epoch_end' event of the Keras Callback class

    # Arguments
        graphs: list with some or all of ('acc', 'loss')
        save_graph: Save graph as an image on Keras Callback 'on_train_end' event 

    """

    def __init__(self, graphs=['loss','R2'], save_graph=False, names ='def', name_model = 'default.png'):
        self.graphs = graphs
        self.num_subplots = len(graphs)
        self.save_graph = save_graph
        self.name_model = name_model
        self.names = names

    def on_train_begin(self, logs={}):
        self.val_R2 = []
        self.R2 = []
        self.val_acc = []
        self.acc = []
        self.loss = []
        self.val_loss = []
        self.epoch_count = 0
        plt.ion()

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_count += 1
        self.val_R2.append(logs.get('val_R2'))
        self.R2.append(logs.get('R2'))
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        epochs = [x for x in range(self.epoch_count)]
        plt.figure(1, figsize=(20,10))
        count_subplots = 0
        if 'acc' in self.graphs:
            count_subplots += 1
            plt.subplot(self.num_subplots, 1, count_subplots)
            plt.title('Accurac [T/V] : [%s/%s] '%( 100*np.around(self.acc[-1],5) , 100*np.around(self.val_acc[-1],5) ) )
            #plt.axis([0,100,0,1])
            plt.plot(epochs, self.val_acc, color='r')
            plt.plot(epochs, self.acc, color='b')
            plt.ylabel('accuracy')
            plt.grid()

            red_patch = mpatches.Patch(color='red', label='Valid')
            blue_patch = mpatches.Patch(color='blue', label='Train')

            plt.legend(handles=[red_patch, blue_patch], loc=4)
        
        if 'R2' in self.graphs:
            count_subplots += 1
            plt.subplot(self.num_subplots, 1, count_subplots)
            plt.title('%s -> R2 [T/V] : [%s / %s] '% 
            (self.names ,np.round(self.R2[-1],5) , np.round(self.val_R2[-1], 5) ))
            #plt.axis([0,100,0,1])
            plt.plot(epochs, self.val_R2, color='r')
            plt.plot(epochs, self.R2, color='b')
            #plt.ylabel('R2')
            plt.grid()

            red_patch = mpatches.Patch(color='red', label='Valid')
            blue_patch = mpatches.Patch(color='blue', label='Train')

            plt.legend(handles=[red_patch, blue_patch], loc=4)
        
        if 'loss' in self.graphs:
            count_subplots += 1
            plt.subplot(self.num_subplots, 1, count_subplots)
            plt.title('%s -> Loss [T/V] : [%s/%s]' %(self.names, np.around(self.loss[-1],5)  , np.around(self.val_loss[-1],5) ))
            #plt.axis([0,100,0,5])
            plt.plot(epochs, self.val_loss, color='r')
            plt.plot(epochs, self.loss, color='b')
            #plt.ylabel('loss')
            plt.grid()

            red_patch = mpatches.Patch(color='red', label='Valid')
            blue_patch = mpatches.Patch(color='blue', label='Train')

            plt.legend(handles=[red_patch, blue_patch], loc=4)

        #plt.draw()
        plt.pause(0.01)

    def on_train_end(self, logs={}):
        plt.figure(1)
        if self.save_graph:
            plt.savefig(self.name_model)
        plt.close(1)

class ConfusionMatrixPlotter(Callback):
    """Plot the confusion matrix on a graph and update after each epoch

    # Arguments
        X_val: The input values 
        Y_val: The expected output values
        classes: The categories as a list of string names
        normalize: True - normalize to [0,1], False - keep as is
        cmap: Specify matplotlib colour map
        title: Graph Title

    """
    def __init__(self, X_val, Y_val, classes, normalize=False, cmap=plt.cm.Blues, title='Confusion Matrix',
                        save_graph=False, name_model = 'default.png'):
        self.X_val = X_val
        self.Y_val = Y_val
        self.title = title
        self.classes = classes
        self.normalize = normalize
        self.save_graph = save_graph
        self.name_model = name_model
        self.cmap = cmap
        plt.ion()
        #plt.show()

        plt.title(self.title)
        
    def on_train_begin(self, logs={}):
        pass
    def on_epoch_end(self, epoch, logs={}):  
        pass

    def on_train_end(self, logs={}):
        plt.figure(2)
        plt.figure(figsize=(20,10))
        pred = self.model.predict(self.X_val)
        max_pred = np.argmax(pred, axis=1)
        max_y = np.argmax(self.Y_val, axis=1)
        cnf_mat = confusion_matrix(max_y, max_pred)
   
        if self.normalize:
            cnf_mat = cnf_mat.astype('float') / cnf_mat.sum(axis=1)[:, np.newaxis]

        thresh = cnf_mat.max() / 2.
        for i, j in itertools.product(range(cnf_mat.shape[0]), range(cnf_mat.shape[1])):
            plt.text(j, i, cnf_mat[i, j],                                          
                         horizontalalignment="center",
                         color="white" if cnf_mat[i, j] > thresh else "black")

        #plt.imshow(cnf_mat, interpolation='nearest', cmap=self.cmap)

        # Labels
        tick_marks = np.arange(len(self.classes))
        plt.xticks(tick_marks, self.classes, rotation=45)
        plt.yticks(tick_marks, self.classes)

        plt.colorbar()
                                                                                                         
        plt.tight_layout()                                                    
        plt.ylabel('True label')                                              
        plt.xlabel('Predicted label')                                         
        #plt.draw()
        #plt.show()
        plt.pause(0.001)   
        if self.save_graph:
            plt.savefig(self.name_model)
        plt.close(2)

class evaluateModel(Callback):
    """Plot the regression in out & out_pred

    # Arguments
        X_val: The input values 
        Y_val: The expected output values
        normalize: True - normalize to [0,1], False - keep as is
        cmap: Specify matplotlib colour map
        title: Graph Title

    """
    def __init__(self, In, Out, names, cmap=plt.cm.Blues, title='Evaluate_model',
                        save_graph=False, name_model = 'Eval_mod.png'):
        self.In = In
        self.Out = Out
        self.title = title
        self.names = names
        self.save_graph = save_graph
        self.name_model = name_model
        self.cmap = cmap

                
    def on_train_begin(self, logs={}):
        self.val_r2_metricKeras = []


    def on_epoch_end(self, epoch, logs={}):  
        
        self.val_r2_metricKeras.append(logs.get('val_r2_metricKeras'))
        
        if (epoch % 10) == 0 :
            plt.figure(3, figsize=(25,20))
            out_pred = self.model.predict(self.In)
            m = np.round( np.mean( np.abs(self.Out - out_pred) ) , 4) 
            

            if(self.Out.shape[-1] == self.In.shape[0]) :
                plt.plot(self.Out, out_pred, '.', markersize = 2)
                plt.title('%s -> R2 =%s / ErrMean = %s '%(self.names, np.round(self.val_r2_metricKeras[-1],5), m), fontsize=10)
                plt.grid(True)
                plt.xlabel = "True values"
                plt.ylabel = "Predict values"
                plt.plot([0,np.max(self.Out)],[0,np.max(self.Out)])
            else :
                c = 0
                fig3 , ax = plt.subplots(nrows=2, ncols=3)
                R2 = tools.np_R2(self.Out, out_pred)
                for i in range (2) :
                    for j in range(3) :
                        ax[i,j].plot(self.Out[:,c], out_pred[:,c], '.', markersize = 2)
                        ax[i,j].set_title('%s / R2 =%s / ErrMean = %s '%(self.names[c], R2[c], m[c]), fontsize=10)
                        ax[i,j].grid(True)
                        ax[i,j].xlabel = "True values"
                        ax[i,j].ylabel = "Predict values"
                        ax[i,j].plot([0,np.max(self.Out[:,c])],[0,np.max(self.Out[:,c])])
                        c +=1
                    plt.draw()
                    plt.pause(0.001)

    def on_train_end(self, logs={}):

        plt.figure(3)
        out_pred = self.model.predict(self.In[range(10000)])
        m = np.round( np.mean( np.abs(self.Out[range(10000)] - out_pred) ) , 4) 
            

        if(self.Out.shape[-1] == self.In.shape[0]) :
            plt.plot(self.Out[range(10000)], out_pred, '.', markersize = 2)
            plt.title('%s -> R2 =%s / ErrMean = %s '%(self.names, np.round(self.val_r2_metricKeras[-1],5), m), fontsize=10)
            plt.grid(True)
            plt.xlabel = "True values"
            plt.ylabel = "Predict values"
            plt.plot([0,np.max(self.Out[range(10000)])],[0,np.max(self.Out[range(10000)])])
        else :
            c = 0
            fig3 , ax = plt.subplots(nrows=2, ncols=3, figsize = (25,20))  
            R2 = tools.np_R2(self.Out, out_pred)

            for i in range (2) :
                for j in range(3) :
                    ax[i,j].plot(self.Out[range(10000),c], out_pred[:,c], '.', markersize = 2)
                    ax[i,j].set_title('%s / R2 =%s / ErrMean = %s '%(self.names[c], R2[c], m[c]), fontsize=10)
                    ax[i,j].grid(True)
                    ax[i,j].xlabel = "True values"
                    ax[i,j].ylabel = "Predict values"
                    ax[i,j].plot([0,np.max(self.Out[range(10000),c])],[0,np.max(self.Out[range(10000),c])])
                    c +=1
                plt.draw()
                plt.pause(0.001)
        
        if self.save_graph:
            plt.savefig(self.name_model)
        plt.close(3)
