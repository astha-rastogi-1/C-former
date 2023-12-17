import scipy.io as sio
import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical

mat_file = sio.loadmat('Corrected_Data/DE_data_all_corrected.mat')
mat_file_labels = sio.loadmat('Corrected_Data/label.mat')
count = 0
accuracy = []

def load_data():
    data = mat_file['data']
    labels = mat_file_labels['label']

    labels = labels[0]
    labels_edited = np.empty(675)
    for i in range(0,45):
        labels_edited[i*15:(i+1)*15] = labels

    data, labels_edited = shuffle(data, labels_edited)
    return data, labels_edited

def process_data():
    data, labels_edited = load_data()
    max = np.max(data)
    min = np.min(data)

    #Normalizing data
    data = data/max
    data = (data - np.mean(data))/np.std(data)

    #Cropping end time segments
    data = data[:,:,0:250,:]

    #Splitting Dataset into train, validation, test 
    train_labels = labels_edited[0:550]
    test_labels = labels_edited[550:600]
    train_data = data[0:550]
    test_data = data[550:600]
    final_test = data[600:675]
    final_labels = labels_edited[600:675]
    cf_labels = np.where(final_labels== -1, 2, final_labels) #Label in the form for confusion matrix
    un, co = np.unique(cf_labels, return_counts=True)
    print(f'Unique: {un}, Counts: {co}')

    train_labels_reshaped = train_labels.reshape(-1,1) #Formatting for input to the CNN model
    test_labels_reshaped = test_labels.reshape(-1,1)
    final_labels_reshaped = final_labels.reshape(-1,1)

    train_labels_reshaped = to_categorical(train_labels_reshaped, 3) #One Hot Encoding
    test_labels_reshaped = to_categorical(test_labels_reshaped, 3)
    final_labels_reshaped = to_categorical(final_labels_reshaped, 3)

    print(train_data.shape)
    return train_data, train_labels_reshaped, test_data, test_labels_reshaped, final_test, final_labels_reshaped