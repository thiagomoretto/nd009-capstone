import matplotlib.pyplot as plt


class TrainPlotter(object):
    """Given a Keras trained model, it provides useful methods to
       plot metrics of testing procedure"""
    def __init__(self, trained_model):
        self.trained_model = trained_model
        
    def plot_history(self, ylim1=None, ylim2=None):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(14, 4))

        plt.subplot(121)
        plt.plot(self.trained_model.train_history.history['acc'],
                 color='salmon', linestyle='dashed', linewidth=2)
        plt.plot(self.trained_model.train_history.history['val_acc'],
                 color='gray', linestyle='dashed', linewidth=2)
        plt.title('Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        if ylim1:
            plt.ylim(ylim1)

        plt.subplot(122)
        plt.plot(self.trained_model.train_history.history['loss'],
                 color='salmon', linestyle='dashed', linewidth=2)
        plt.plot(self.trained_model.train_history.history['val_loss'],
                 color='gray', linestyle='dashed', linewidth=2)
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        if ylim2:
            plt.ylim(ylim2)
        plt.show()
        
        
def plot_compare(model1, model2, legend1, legend2):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 4))

    plt.subplot(121)
    
    plt.plot(model1.train_history.history['acc'],
             color='salmon',
             linestyle='dashed',
             label="{} train".format(legend1))
    plt.plot(model1.train_history.history['val_acc'],
             color='salmon',
             linestyle='dotted',
             label="{} validation".format(legend1),
             alpha=.5)
    
    plt.plot(model2.train_history.history['acc'],
             color='gray',
             linestyle='dashed',
             label="{} train".format(legend2))
    plt.plot(model2.train_history.history['val_acc'],
             color='gray',
             label="{} validation".format(legend2),
             linestyle='dotted', alpha=.5)
    
    plt.title('Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')

    plt.subplot(122)
    
    plt.plot(model1.train_history.history['loss'],
             color='salmon',
             linestyle='dashed',
             label="{} train".format(legend1))
    plt.plot(model1.train_history.history['val_loss'],
             color='salmon',
             linestyle='dotted',
             label="{} validation".format(legend1),
             alpha=.5)
    
    plt.plot(model2.train_history.history['loss'],
             color='gray',
             linestyle='dashed',
             label="{} train".format(legend2))
    plt.plot(model2.train_history.history['val_loss'],
             color='gray',
             label="{} validation".format(legend2),
             linestyle='dotted', alpha=.5)
    
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.show()
    
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import numpy as np
    import itertools
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure(dpi=120)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
def display_image(img_path, title=None):
    import cv2
    import matplotlib.pyplot as plt
    img = cv2.imread(img_path)
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(cv_rgb)
    if title:
        plt.title(title)
    plt.show()  