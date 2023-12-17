import matplotlib as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from tensorflow import keras 


def metrics(final_test, final_labels_reshaped, vit_classifier):
    # Reshaping the labels for the various metrics
    final_labels = np.argmax(final_labels_reshaped, axis=1) # labels are converted to 0,1,2 from one hot encoding
    cf_labels = np.where(final_labels== -1, 2, final_labels) # Replace any label -1 with value 2 (why?)

    #Calculating Metrics, Confusion Matrices and Classification Reports
    final_preds = np.empty([3, 75])

    his_exp = vit_classifier.evaluate(x = final_test, y = final_labels_reshaped)
    accuracy_exp = his_exp[1]
    pred_labels_exp = np.argmax(vit_classifier.predict(final_test), axis=-1)

    final_preds[1,:] = pred_labels_exp


    cf_matrix = sklearn.metrics.confusion_matrix(cf_labels, pred_labels_exp, labels = [2,0,1])

    print(cf_matrix)

    # np.save("/content/drive/MyDrive/EEG/cf_94_3_precision.npy", cf_matrix)
    # # cf_ensemble = sklearn.metrics.confusion_matrix(cf_labels, ensemble_predictions, labels = [2,0,1])
    cf = sns.heatmap(cf_matrix/np.sum(cf_matrix, axis=0), annot=True,
                annot_kws={"size": 40}, cmap='Blues')
    cf.set_xticklabels(cf.get_xmajorticklabels(), fontsize = 30)
    cf.set_yticklabels(cf.get_xmajorticklabels(), fontsize = 30)
    plt.ylabel('True label',fontsize = 30)
    plt.xlabel('Predicted label',fontsize = 30)
    # # sns.set(font_scale=1.4)
    results_path = '/content/drive/MyDrive/EEG/cf_matrix.png'
    plt.savefig(results_path, dpi=400)

    cr_exp = sklearn.metrics.classification_report(cf_labels, pred_labels_exp, labels = [2,0,1], output_dict=True)
    df = pd.DataFrame(cr_exp).transpose()
    print(df)
    df.to_csv('classification_report.csv')

    print("\n", "Accuracy ViT: " + str(accuracy_exp), "\n", "Confusion Matrix ViT: ", str(cf_matrix), "\n", "Classification Report ViT: ", str(cr_exp), sep= '\n')
    # print(f'MCC ViT: {mcc_exp}\n MCC Ensemble: {mcc_ensemble}')

if "__name__"=="__main__":
    vit_classifier = keras.models.load_model("saved_model/best_vit_model_96")
    vit_classifier.load_weights("saved_model/vit_best_96.h5")
    final_test = np.load('saved_model/test_96.npy')
    final_labels_reshaped = np.load("saved_model/test_label_96.npy") # The labels are one hot encoded

    # Show metrics
    metrics(final_test, final_labels_reshaped, vit_classifier)