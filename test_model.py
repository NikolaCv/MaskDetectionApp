from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
import os
from nn_train import generator_from_folder, resize_image
from sklearn import metrics
import numpy as np

if __name__ == '__main__':

    model = keras.models.load_model('nn_model')

    test_data = 'test_images'
    # test_loss, test_acc = model.evaluate(generator_from_folder(test_data, 300, False), verbose=1)
    #
    # print(f'\nTest accuracy: {test_acc}\tTest loss: {test_loss}')

    actual = []
    predictions = []
    for ind, filename in enumerate(os.listdir(test_data)):
        img = cv2.imread(os.path.join(test_data, filename))
        # plt.figure()
        # plt.imshow(img)
        # plt.show()
        print(ind)
        if img is not None:
            info = filename.split('_')
            actual.append(int(info[1]) - 1)
            prediction = model.predict(np.reshape(resize_image(img, 300), (1, 300, 300, 3)),
                                       batch_size=None, verbose=0, steps=None, callbacks=None,
                                       workers=1, use_multiprocessing=False).round(0)
            predictions.append(np.argmax(prediction, axis=1)[0])
            if actual[-1] == predictions[-1]:
                cv2.imwrite(f'{actual[-1] + 11}/{predictions[-1] + 1} {ind}.jpg', img)


    confusion_matrix = metrics.confusion_matrix(actual, predictions)
    print(actual, predictions, confusion_matrix)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                                display_labels=['Correct', 'Under nose', 'Under chin',  'No mask'])

    cm_display.plot(cmap='YlGn')
    plt.show()

    confusion_matrix = np.divide(confusion_matrix, 11236/4)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                                display_labels=['Correct', 'Under nose', 'Under chin',  'No mask'])

    cm_display.plot(cmap='YlGn')
    plt.show()
