"""

    Feature : Traffic_Sign_Classifer
    Author : lockeve11
    Data : 2022/11/24
    Template : Udacity

------------------------------------------

    Frame : Tensorflow

"""
from tqdm import tqdm
import cv2
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Dropout,Rescaling,InputLayer
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau

import wandb
from wandb.keras import WandbCallback

import pandas as pd

index_to_class = {}
class_csv = pd.read_csv('signnames.csv')
for classid,signname in zip(class_csv.ClassId,class_csv.SignName):
    index_to_class.update({classid:signname})

def evaluate_model(model,dataset:list,stage,type,log_file):
    # test_images,test_labels,verbose=2
    loss,acc = model.evaluate(dataset[0],dataset[1],verbose=2)
    print('\n')
    print(f'{stage} Model {type} loss: {loss}')
    print(f'{stage} Model {type} accuracy:{acc:.2%}')
    log_file.write('\n')
    log_file.write(f'{stage} Model {type} loss: {loss}\n')
    log_file.write(f'{stage} Model {type} accuracy:{acc:.2%}\n')
    return loss, acc

def infer_analysis(model,dataset:list,infer_type,sequence,index_to_class,root,log_file,save_img=True):
    pred_folder = sequence + infer_type
    pred_folder_path = os.path.join(root,pred_folder)
    label_list = [label for label in index_to_class.values()]
    if not os.path.exists(pred_folder_path):
        os.mkdir(pred_folder_path)
    pred_correct_path = os.path.join(pred_folder_path, 'correct')
    if not os.path.exists(pred_correct_path):
        os.mkdir(pred_correct_path)
    pred_wrong_path = os.path.join(pred_folder_path, 'wrong' )
    if not os.path.exists(pred_wrong_path):
        os.mkdir(pred_wrong_path)

    class_total = {k:0 for k in index_to_class.keys()}
    correct_count = {k:[0,0] for k in index_to_class.keys()}

    print(f'correct_count {correct_count}')

    y_pred = []
    y_true = []
    wrong_count = []
    img_count = 0

    for image,label in zip(dataset[0],dataset[1]):
        img_count += 1
        image_copy = image.copy()
        image_copy = cv2.resize(image_copy,[400,400])
        onebach_image = np.expand_dims(image,axis=0)
        img_infer = model.predict(onebach_image)
        img_pred = int(np.argmax(img_infer))
        y_pred.append(img_pred)
        conf = round(np.squeeze(img_infer)[img_pred],2)
        true_label = label
        y_true.append(true_label)
        class_total[true_label] +=1
        #image = image[0].astype('uint8')
    
        img_text_Origin = 'Origin:' + index_to_class[true_label]
        cv2.putText(image_copy,img_text_Origin,(0,50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
        img_text_Predict = 'Predict:' + index_to_class[img_pred] 
        cv2.putText(image_copy,img_text_Predict,(0,150),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
        img_text_Conf = 'confid' + str(conf)
        cv2.putText(image_copy,img_text_Conf,(0,200),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

        if img_pred == true_label:
            correct_count[img_pred][0] += 1
            correct_count[img_pred][1] += conf
            img_save_path = os.path.join(pred_correct_path,str(img_count) + '.jpg')
            if save_img:
                cv2.imwrite(img_save_path,image_copy)
        else:
            wrong_count.append(conf)
            img_save_path = os.path.join(pred_wrong_path, str(img_count)+'.jpg')
            if save_img:
                cv2.imwrite(img_save_path, image_copy)
        
    log_file.write('\n')
    log_file.write(f'total images: {img_count}\n')

    matrix_shape = len(index_to_class)
    confusion_matrix = np.zeros((matrix_shape,matrix_shape))

    for item in zip(y_pred,y_true):
        confusion_matrix[item[0],item[1]] += 1
    log_file.write(f'hang is prediction, lie is true label\n')
    log_file.write(f'{label_list}\n')
    log_file.write('confusion matrix is:  \n')
    for hang in confusion_matrix:
        log_file.write(f'{hang}\n')
    
    pred_sum = np.sum(confusion_matrix, axis=1)
    label_sum = np.sum(confusion_matrix, axis=0)

    for index in index_to_class.keys():
        if pred_sum[index] != 0 :
            precision = confusion_matrix[index][index] / pred_sum[index]
            precision = format(precision,'.2%')
        else:
            precision = 'None,no prediciotn in this category'
        
        if label_sum[index] != 0:
            recall = confusion_matrix[index][index] / label_sum[index]
            recall = format(recall,'.2%')
        else:
            recall = 'None,no data in this category'

    print(f'{index_to_class[index]} category accuracy/recall is {recall}, precision is {precision}')
    log_file.write(f'{index_to_class[index]} category accuracy/recall is {recall}, precision is {precision}\n')

    avg_acc = sum([item[0] for item in correct_count.values()]) / img_count
    print(f'average classification accuracy is {avg_acc:.2%}')
    log_file.write(f'average classification accuracy is {avg_acc:.2%}\n')

    if sum([item[0] for item in correct_count.values()]) != 0:
        avg_correct_conf = sum([item[1] for item in correct_count.values()]) / sum(
        [item[0] for item in correct_count.values()])
        print(f'average correct classification conf is {round(avg_correct_conf, 2)}')
        log_file.write(f'average correct classification conf is {round(avg_correct_conf, 2)}\n')
    else:
        print(f'no correct classification in current dataset')
        log_file.write(f'no correct classification in current dataset')




# ******************************************

# ---------Function : LOAD DATA-------------

# Notice : because of format of dataset,Need we use different way to load data.

curr_root = os.getcwd()

training_file = os.path.join(curr_root,'traffic-signs-data','train.p')
validation_file = os.path.join(curr_root,'traffic-signs-data','valid.p')
testing_file = os.path.join(curr_root,'traffic-signs-data','test.p')

with open(training_file,mode='rb') as f:
    train = pickle.load(f)
with open(validation_file,mode="rb") as f:
    valid = pickle.load(f)
with open(testing_file,mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
y_test_copy = test['labels']

# # Summary Dataset 

# """

# - `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
# - `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
# - `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
# - `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**

# """

n_train = len(X_train)
n_validataion = len(X_valid)
n_test = len(X_test)
image_shape = X_train[0].shape
n_classes = len(set(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

index = np.random.randint(n_train)
train_image_vis = X_train[index]
# plt.imshow(train_image_vis)
print("y_train[index] = {}".format(y_train[index]))

# Counter : return an iterable object
vis_count_train = Counter(y_train)
range_x = np.array(range(n_classes))
range_y = [vis_count_train[i] for i in range_x]
plt.figure(figsize=(9,5))
plt.bar(range_x,range_y)
plt.xticks(list(range(n_classes)))
plt.xlabel('class')
plt.ylabel('numbers')
plt.title('Train Data Distribution')
# plt.show()


# ******************************************

# ---------Function : Model Architecture-------------
#               Build Model
# Notice:In the project, we use pre-trained model.
#        Before this,we must pre-process Dataset.

def grayscale(image):
    resize_image = cv2.resize(image,[32,32])
    # gray = cv2.cvtColor(resize_image,cv2.COLOR_RGB2GRAY)
    # image = tf.image.grayscale_to_rgb(gray)
    # after gary the shape of image may be (224,224) but we need (224,224,1)
    # return np.expand_dims(gray,axis=2)
    return resize_image

def preprocess(images):
    list1 = []
    for image in images:
        list1.append(grayscale(image))
    list1 = np.array(list1)
    return list1

def one_hot(images):
    list1 = np.zeros((len(images),n_classes))
    for i,label in enumerate(images):
        list1[i][label] = 1
    return list1

X_train,y_train = shuffle(X_train,y_train)
# X_train = preprocess(X_train)
y_train = one_hot(y_train)
# X_valid = preprocess(X_valid)
y_valid = one_hot(y_valid)
# X_test = preprocess(X_test)
y_test = one_hot(y_test)

# ******************************************

# ----------    Parameters Setting    ------------

save_best_loss_weights_folder = "best_loss_weights"
best_loss_weights_path = os.path.join(curr_root,save_best_loss_weights_folder)
if not os.path.exists(best_loss_weights_path):
    os.mkdir(best_loss_weights_path)
print('best_loss_weights_path is ',best_loss_weights_path)
save_last_epoch_folder = "last_loss_weights"
last_epoch_weights_path = os.path.join(curr_root,save_last_epoch_folder)
if not os.path.exists(last_epoch_weights_path):
    os.mkdir(last_epoch_weights_path)
print('last_epoch_weights_path is ',last_epoch_weights_path)

wandb.init(project="MobileNet_Traffic_Sign")
config = wandb.config
config.learning_rate = 0.0001
config.batch_size = 16
config.epochs = 10
config.classes = n_classes
stopping_patience = 5
version_sequnce = 1_1

image_size = 32

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(mode='horizontal',input_shape=(image_size,image_size,3)),
        tf.keras.layers.RandomRotation(0.022)
    ]
)

# inputs = InputLayer(input_shape=(image_size,image_size,3),name="input")
# x = Rescaling(1./128)(inputs)
base_model = MobileNet(input_shape=(image_size,image_size,3),weights="imagenet",include_top=False)
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dropout(0.3)(x)
# prediction = Dense(config.classes,activation="softmax",name="dense")
# model = tf.keras.Model(inputs=inputs,outputs=prediction)

# model = Sequential([
#     InputLayer(input_shape=(image_size,image_size,3),name="input_1"),
#     #data_augmentation,
#     Rescaling(1./128),
#     base_model,
#     GlobalAveragePooling2D(),
#     Dropout(0.3),
#     Dense(config.classes,activation="softmax",name="dense")
# ])

# print(model.summary())


# ******************************************
 
# ---------Function : Model Train Valid -------------

# opt = tf.keras.optimizers.Nadam(config.learning_rate)
# acc = tf.keras.metrics.CategoricalAccuracy()
# model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=[acc])


# #Available metrics are: loss,categorical_accuracy,val_loss,val_categorical_accuracy
# checkpointer_loss = ModelCheckpoint(filepath=best_loss_weights_path,monitor='val_loss',save_best_only=True,verbose=1,mode='min')
# earlystopper=EarlyStopping(monitor='val_categorical_accuracy',min_delta=0.001,patience=stopping_patience,verbose=1,mode='max')
# reduce_lr = ReduceLROnPlateau(monitor='val_categorical_accuracy',factor=0.8,patience=6, verbose=1, mode='max', min_delta=0.001, min_lr=0.0004)

# mobilenet_history = model.fit(X_train,y_train,epochs=config.epochs,verbose=1,validation_data=[X_valid,y_valid],
#                     callbacks=[checkpointer_loss,earlystopper,reduce_lr,WandbCallback()])
                    
# model.save(last_epoch_weights_path)
# print(f'save last epoch model at {last_epoch_weights_path}')



# ******************************************

# ---------Function : Evaluate Model -------------

logfile_name = "./" + str(version_sequnce) + "_logfile.txt"
log_file = open(logfile_name,'w',encoding='utf-8')
print(f'{logfile_name} is saved at current foloder {curr_root}')
log_file.write(f'save best val loss model at {best_loss_weights_path}')
log_file.write(f'save last val loss model at {last_epoch_weights_path}')
log_file.write('\n')

log_file.write(f'testset num is :{n_test}')

model_load_lastepoch = tf.keras.models.load_model(last_epoch_weights_path)

valid_lastepoch_loss,valid_lastepoch_acc = evaluate_model(model_load_lastepoch,[X_valid,y_valid],'last epoch','valid',log_file)
test_lastepoch_loss,test_lastepoch_acc = evaluate_model(model_load_lastepoch,[X_test,y_test],'last epoch','test',log_file)

model_load_bestloss = tf.keras.models.load_model(best_loss_weights_path)
valid_best_loss,valid_best_acc = evaluate_model(model_load_bestloss,[X_valid,y_valid],'best epoch','valid',log_file)
test_best_loss,test_best_acc = evaluate_model(model_load_bestloss,[X_test,y_test],'best epoch','test',log_file)

model_test_metrics_list = [(test_lastepoch_acc,'lastacc'),(test_best_acc,'bestacc')]
model_test_metrics_list.sort(reverse=True)
best_acc = model_test_metrics_list[0][1]

if best_acc == 'lastacc':
    model_analysis = model_load_lastepoch
    log_file.write('\n')
    log_file.write('use last epoch model to analysis the dataset\n')
else:
    model_analysis = model_load_bestloss
    log_file.write('\n')
    log_file.write('use best lost epoch model to analysis the dataset\n')


# ******************************************

# ---------Function : Infer Analysis on test dataset -------------

infer_analysis(model_analysis,[X_test,y_test_copy],'test',str(version_sequnce),index_to_class,curr_root,log_file)

log_file.close()




