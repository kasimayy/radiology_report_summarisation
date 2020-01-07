import matplotlib.pyplot as plt

plt.style.use('ggplot')

def plot_history(history):
    if 'acc' in history.keys():
        acc = history['acc']
        val_acc = history['val_acc']

    elif 'accuracy' in history.keys():
        acc = history['accuracy']
        val_acc = history['val_accuracy']
    else:
        acc = []
        val_acc = []
    loss = history['loss']
    val_loss = history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Accuracy')
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Loss')
    plt.legend()
    
    if 'recall' in history.keys():
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 3, 1)
        recall = history['recall']
        val_recall = history['val_recall']
        plt.plot(x, recall, 'b', label='Training recall')
        plt.plot(x, val_recall, 'r', label='Validation recall')
        plt.title('Recall')
        plt.legend()
        
    if 'precision' in history.keys():
        plt.subplot(1, 3, 2)
        specificity = history['precision']
        val_specificity = history['val_precision']
        plt.plot(x, specificity, 'b', label='Training precision')
        plt.plot(x, val_specificity, 'r', label='Validation precision')
        plt.title('Precision')
        plt.legend()
        
    if 'binary_accuracy' in history.keys():
        plt.subplot(1, 3, 3)
        specificity = history['binary_accuracy']
        val_specificity = history['val_binary_accuracy']
        plt.plot(x, specificity, 'b', label='Training binary accuracy')
        plt.plot(x, val_specificity, 'r', label='Validation binary accuracy')
        plt.title('Binary Accuracy')
        plt.legend()
    
    
    