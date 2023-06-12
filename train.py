import traceback
import h5py
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.python.keras.layers import Multiply, Permute
import tensorflow.keras.backend as K


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        # # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


def generate_model(model_shape):
    weight_decay = 1e-3

    # Attention
    input_layer = Input(shape=model_shape)

    cnn = Conv2D(input_layer.shape[3], (3, 3), padding='same', activation='relu')(input_layer)
    cnn = Conv2D(input_layer.shape[3], (3, 3), padding='same', activation='relu')(cnn)

    a = Permute((1, 3, 2))(cnn)
    a = Dense(cnn.shape[2], activation='sigmoid')(a)
    a_probs = Permute((1, 3, 2), name='attention_vec')(a)

    output_atteintion_mul = Multiply()([cnn, a_probs])

    cnn_2 = Conv2D(input_layer.shape[3] * 2, (3, 3), padding='same', activation='relu')(output_atteintion_mul)
    cnn_2 = Conv2D(input_layer.shape[3] * 2, (3, 3), padding='same', activation='relu')(cnn_2)

    a_2 = Permute((1, 3, 2))(cnn_2)
    a_2 = Dense(cnn_2.shape[2], activation='sigmoid')(a_2)
    a_probs_2 = Permute((1, 3, 2), name='attention_vec_2')(a_2)

    output_atteintion_mul_2 = Multiply()([cnn_2, a_probs_2])
    max_pool_2 = MaxPooling2D()(output_atteintion_mul_2)
    dropout_2 = Dropout(0.25)(max_pool_2)

    outpus = Flatten()(dropout_2)
    outpus = Dropout(0.5)(outpus)

    outputs = Dense(1, activation='sigmoid')(outpus)

    model = Model(inputs=input_layer, outputs=outputs, name='my_model')
    model._name = 'my_model'
    model.summary()

    return model


def getdata(path, model='ensemble'):
    f = h5py.File(path, 'r')
    healthy_data = f['healthy_data']
    healthy_label = f['healthy_label']
    seizure_data = f['seizure_data']
    seizure_label = f['seizure_label']
    healthy_data, healthy_label, seizure_data, seizure_label = np.array(healthy_data), np.array(healthy_label), \
                                                               np.array(seizure_data), np.array(seizure_label)

    if len(seizure_data) == 0:
        data = healthy_data
        label = healthy_label
    else:
        data = np.concatenate((healthy_data, seizure_data))
        label = np.concatenate((healthy_label, seizure_label))

    return data, label


model_name = ''
patient_name = ''
kfold = 0
patience = 5
lr = 1e-3

def scheduler(epoch):
    return 1e-3

def train(train_patient_list, test_patient, time, mode, train_files, test_files, k, epoch, batch_size):
    dir_path = '.\\model\\{patient}\\'.format(model=mode, time=time, patient=test_patient)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


    print('[TRAIN] train_patient_list, test_patient, mode, time > ', train_patient_list, test_patient, mode, time, 'start..')
    df = pd.DataFrame()

    train_path = '.\\preprocess\\{patient}\\{file}'

    # Load and train the rest of the data except for test_patient
    for round in range(len(test_files)):
        X_train = None
        Y_train = None

        except_cnt = 0
        for round_2 in range(len(train_files)):
            # print('train_files[' + str(round_2) + '] :' + str(train_files[round_2]))

            temp_X_train, temp_Y_train = getdata(
                train_path.format(patient=train_patient_list[round_2], file=train_files[round_2][0]), model=mode)

            if len(train_files[round_2]) != 1:
                for train_file in train_files[round_2][1:]:
                    Xt, Yt = getdata(train_path.format(time=time, patient=train_patient_list[round_2], file=train_file), model=mode)
                    temp_X_train = np.concatenate((temp_X_train, Xt))
                    temp_Y_train = np.concatenate((temp_Y_train, Yt))

            # concatenate
            if X_train is None:
                X_train = temp_X_train
                Y_train = temp_Y_train
            else:
                X_train = np.concatenate((X_train, temp_X_train))
                Y_train = np.concatenate((Y_train, temp_Y_train))

            # print('round ' + str(round_2) + 'X_train.shape[1:] :' + str(X_train.shape[1:]))

        Y_train = np.argmax(Y_train, axis=1)


        # Ensemble
        X_train = np.reshape(X_train, (X_train.shape[0], 60, 31, 18))
        my_model = generate_model(X_train.shape[1:])

        tf.keras.utils.plot_model(my_model,
                                  to_file='model/model_plot.png',
                                  show_shapes=True,
                                  )
        model_list = [my_model]


        g_train = None
        g_validation = None
        target_model = None

        for t_model in model_list:
            skf = KFold(n_splits=k)
            global model_name, patient_name, kfold
            model_name = t_model.name
            patient_name = test_patient
            kfold = 0

            for train, validation in skf.split(X_train):
                # print('train', train, len(train))
                # print('validation', validation, len(validation))
                # print('Y_train', Y_train[train])

                kfold += 1

                target_model = generate_model(X_train.shape[1:])

                for layer in target_model.layers:
                    layer._name = layer.name + '_' + str(kfold)

                optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
                METRICS = [
                    tf.keras.metrics.TrueNegatives(name='tn'),
                    tf.keras.metrics.FalsePositives(name='fp'),
                    tf.keras.metrics.FalseNegatives(name='fn'),
                    tf.keras.metrics.TruePositives(name='tp'),
                    tf.keras.metrics.SensitivityAtSpecificity(specificity=0.1, num_thresholds=1, name='sen'),
                    tf.keras.metrics.SpecificityAtSensitivity(sensitivity=0.1, num_thresholds=1, name='spe'),
                    tf.keras.metrics.Precision(name='pre'),
                    ['accuracy']
                ]

                target_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=METRICS)

                ckpt = dir_path
                ckpt = ckpt + target_model.name + '_' + str(kfold) + '_' + test_patient + '.h5'

                global patience
                cb = [ModelCheckpoint(filepath=ckpt, monitor='val_loss', save_best_only=True, save_weights_only=True,
                                      mode='min', period=1, verbose=0),
                      tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience),
                      tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)]

                neg = 0
                pos = 0
                don = 0
                for x in Y_train[train]:
                    if x == 0:
                        neg += 1
                    elif x == 1:
                        pos += 1
                    else:
                        don += 1
                # print('[train] neg pos don', neg, pos, don)
                total = neg + pos

                val_neg = 0
                val_pos = 0
                val_don = 0
                for val_x in Y_train[validation]:
                    # print('x[0] [1]', x[0], x[1])
                    if val_x == 0:
                        val_neg += 1
                    elif val_x == 1:
                        val_pos += 1
                    else:
                        val_don += 1
                # print('[validation] neg pos don', val_neg, val_pos, val_don)

                weight_for_neg = (1 / neg) * total / 2.0
                weight_for_pos = (1 / pos) * total / 2.0
                class_weights = {0: weight_for_neg, 1: weight_for_pos}

                history = target_model.fit(X_train[train], Y_train[train], epochs=epoch, batch_size=batch_size,
                                           validation_data=(X_train[validation], Y_train[validation]), verbose=1,
                                           callbacks=cb, class_weight=class_weights)

                # k fold
                # Y_label = np.argmax(Y_train[validation], axis=1)
                Y_label = Y_train[validation]

                ## Load model
                target_model.load_weights(ckpt)

                eval = target_model.evaluate(X_train[validation], Y_train[validation])
                print("\nVAL LOSS: %.4f ACCURACY: %.4f" % (eval[0], eval[8]))

                X_pred = target_model.predict(X_train[validation])

                Y_pred = []
                for pred in X_pred:
                    if pred[0] <= 0.5:
                        Y_pred.append(0)
                    else:
                        Y_pred.append(1)

                tn, fp, fn, tp = confusion_matrix(Y_label, Y_pred).ravel()
                print('VAL kfold, test_patient, tn, fp, fn, tp:', kfold, test_patient, tn, fp, fn, tp)
                sen = None
                spe = None
                ppv = None
                fpr = None
                if tp == 0 or (tp + fn) == 0:
                    sen = 0
                else:
                    sen = tp / (tp + fn)

                if tn == 0 or (tn + fp) == 0:
                    spe = 0
                else:
                    spe = tn / (tn + fp)

                if tp == 0 or (tp + fp) == 0:
                    ppv = 0
                else:
                    ppv = tp / (tp + fp)

                if fp == 0 or (fp + tn) == 0:
                    fpr = 0
                else:
                    fpr = fp / (fp + tn)
                acc = (tp + tn) / (tp + tn + fp + fn)
                print('sen spe ppv fpr acc', sen, spe, ppv, fpr, acc)

                del target_model, Y_pred
                gc.collect()
                tf.keras.backend.clear_session()

                g_train = train
                g_validation = validation

        print('X_train', X_train.shape)
        print('Y_train', Y_train.shape)

        del X_train, Y_train


test_result = []


def test(train_patient_list, test_patient, time, mode, test_files, k):
    if test_patient == 'chb16':
        return

    # Cross Patient Specific Method
    global test_result


    print('[TEST] train_patient_list, test_patient, mode, time > ', train_patient_list, test_patient, mode, time, 'start..')

    df = pd.DataFrame()

    path = '.\\preprocess\\{test_patient}\\{file}'

    # predict
    for round_ in range(len(test_files)):

        try:
            print('test_files[' + str(round_) + ']:' + str(test_files[round_]))

            temp_df = []
            X_test = None
            Y_test = None

            # load test_files
            for round_2 in range(len(test_files)):

                X_test, Y_test = getdata(
                    path.format(test_patient=test_patient, file=test_files[round_2][0]), model=mode)

                if len(test_files[round_2]) != 1:
                    for test_file in test_files[round_2][1:]:
                        Xt, Yt = getdata(
                            path.format(time=time, test_patient=test_patient, file=test_file), model=mode)
                        X_test = np.concatenate((X_test, Xt))
                        Y_test = np.concatenate((Y_test, Yt))

            print('len(X_test):', len(X_test))
            print('len(Y_test):', len(Y_test))

            X_test = np.reshape(X_test, (X_test.shape[0], 60, 31, 18))

            Y_test = np.argmax(Y_test, axis=1)
            # Y_label = np.argmax(Y_test, axis=1)

            ckpt = '.\\model\\{patient}\\'.format(time=time, patient=test_patient)

            Y_pred_list = []
            Y_pred_avg_list = []
            for i in range(k):
                x_model = generate_model(X_test.shape[1:])
                x_model.load_weights(ckpt + x_model.name + '_' + str(i + 1) + '_' + test_patient + '.h5')

                for layer in x_model.layers:
                    layer._name = layer.name + '_' + str(i)

                optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
                METRICS = [
                    tf.keras.metrics.TrueNegatives(name='tn'),
                    tf.keras.metrics.FalsePositives(name='fp'),
                    tf.keras.metrics.FalseNegatives(name='fn'),
                    tf.keras.metrics.TruePositives(name='tp'),
                    tf.keras.metrics.SensitivityAtSpecificity(specificity=0.1, num_thresholds=1, name='sen'),
                    tf.keras.metrics.SpecificityAtSensitivity(sensitivity=0.1, num_thresholds=1, name='spe'),
                    tf.keras.metrics.Precision(name='pre'),
                    ['accuracy']
                ]

                x_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=METRICS)

                print('kFold', (i + 1))
                eval = x_model.evaluate(X_test, Y_test)
                print("\nAlexNet VAL LOSS: %.4f ACCURACY: %.4f" % (eval[0], eval[8]))
                X_pred = x_model.predict(X_test)
                Y_pred = []
                Y_pred_avg = []
                for pred in X_pred:
                    if pred[0] <= 0.5:
                        Y_pred.append(0)
                    else:
                        Y_pred.append(1)

                    # avg 0728
                    Y_pred_avg.append(pred)

                Y_pred_list.append(Y_pred)
                Y_pred_avg_list.append(Y_pred_avg)

                # Y_label = np.argmax(Y_test, axis=1)
                Y_label = Y_test
                tn, fp, fn, tp = confusion_matrix(Y_label, Y_pred).ravel()
                print('CNN kfold, test_patient, tn, fp, fn, tp :', kfold, test_patient, tn, fp, fn, tp)
                sen = None
                spe = None
                ppv = None
                fpr = None
                if tp == 0 or (tp + fn) == 0:
                    sen = 0
                else:
                    sen = tp / (tp + fn)

                if tn == 0 or (tn + fp) == 0:
                    spe = 0
                else:
                    spe = tn / (tn + fp)

                if tp == 0 or (tp + fp) == 0:
                    ppv = 0
                else:
                    ppv = tp / (tp + fp)

                if fp == 0 or (fp + tn) == 0:
                    fpr = 0
                else:
                    fpr = fp / (fp + tn)
                acc = (tp + tn) / (tp + tn + fp + fn)
                print('sen spe ppv fpr acc', sen, spe, ppv, fpr, acc)

                # 0730
                f1_micro = f1_score(Y_label, Y_pred, average='micro')
                f1_macro = f1_score(Y_label, Y_pred, average='macro')
                roc_auc_micro = roc_auc_score(Y_label, Y_pred, average='micro')
                roc_auc_macro = roc_auc_score(Y_label, Y_pred, average='macro')
                temp_df.append([str(train_patient_list), test_patient, mode, time, str((i + 1)), ckpt, tn, fp, fn, tp,
                                str(round((sen * 100.0), 0)),
                                spe, ppv, str(round(fpr, 3)), acc, f1_micro, roc_auc_micro, f1_macro, roc_auc_macro])

                print()

                del x_model
                gc.collect()
                tf.keras.backend.clear_session()

            # 0, 1
            Y_pred_result_list = []
            for i in range(len(Y_pred_list[0])):
                neg_cnt = 0
                pos_cnt = 0

                for j in range(len(Y_pred_list)):         # k = 5
                    if Y_pred_list[j][i] == 0:
                        neg_cnt += 1
                    elif Y_pred_list[j][i] == 1:
                        pos_cnt += 1

                output_list = []
                if neg_cnt > pos_cnt:
                    output_list.append(0)
                else:
                    output_list.append(1)

                Y_pred_result_list.append(output_list)

            # avg
            Y_pred_avg_result_list = []
            for i in range(len(Y_pred_avg_list[0])):
                avg_list = []
                for j in range(len(Y_pred_avg_list)):
                    avg_list.append(Y_pred_avg_list[j][i])

                avg = sum(avg_list, 0.0) / len(avg_list)
                results = 0
                if avg <= 0.5:
                    results = 0
                else:
                    results = 1

                Y_pred_avg_result_list.append(results)


            # Voting (Ensemble)
            pred_label = np.array(Y_pred_result_list)
            # Y_label = np.argmax(Y_test, axis=1)
            Y_label = Y_test
            tn, fp, fn, tp = confusion_matrix(Y_label, pred_label).ravel()
            print('Ensemble Vote VAL kfold, test_patient, tn, fp, fn, tp :', kfold, test_patient, tn, fp, fn, tp)
            sen = None
            spe = None
            ppv = None
            fpr = None
            if tp == 0 or (tp + fn) == 0:
                sen = 0
            else:
                sen = tp / (tp + fn)

            if tn == 0 or (tn + fp) == 0:
                spe = 0
            else:
                spe = tn / (tn + fp)

            if tp == 0 or (tp + fp) == 0:
                ppv = 0
            else:
                ppv = tp / (tp + fp)

            if fp == 0 or (fp + tn) == 0:
                fpr = 0
            else:
                fpr = fp / (fp + tn)
            acc = (tp + tn) / (tp + tn + fp + fn)
            print('sen spe ppv fpr acc', sen, spe, ppv, fpr, acc)
            test_result.append([sen, fpr, acc])

            f1_micro = f1_score(Y_label, Y_pred, average='micro')
            f1_macro = f1_score(Y_label, Y_pred, average='macro')
            roc_auc_micro = roc_auc_score(Y_label, Y_pred, average='micro')
            roc_auc_macro = roc_auc_score(Y_label, Y_pred, average='macro')

            temp_df.append([str(train_patient_list), test_patient, mode, time, 'Hard Voting', ckpt, tn, fp, fn, tp,
                            str(round((sen * 100), 0)),
                            spe, ppv, str(round(fpr, 3)), acc, f1_micro, roc_auc_micro, f1_macro, roc_auc_macro])



            # Average
            pred_label = np.array(Y_pred_avg_result_list)
            tn, fp, fn, tp = confusion_matrix(Y_label, pred_label).ravel()
            print('Ensemble Avg VAL kfold, test_patient, tn, fp, fn, tp :', kfold, test_patient, tn, fp, fn, tp)

            sen = None
            spe = None
            ppv = None
            fpr = None
            if tp == 0 or (tp + fn) == 0:
                sen = 0
            else:
                sen = tp / (tp + fn)

            if tn == 0 or (tn + fp) == 0:
                spe = 0
            else:
                spe = tn / (tn + fp)

            if tp == 0 or (tp + fp) == 0:
                ppv = 0
            else:
                ppv = tp / (tp + fp)

            if fp == 0 or (fp + tn) == 0:
                fpr = 0
            else:
                fpr = fp / (fp + tn)
            acc = (tp + tn) / (tp + tn + fp + fn)
            print('sen spe ppv fpr acc', sen, spe, ppv, fpr, acc)
            test_result.append([sen, fpr, acc])

            f1_micro = f1_score(Y_label, Y_pred, average='micro')
            f1_macro = f1_score(Y_label, Y_pred, average='macro')
            roc_auc_micro = roc_auc_score(Y_label, Y_pred, average='micro')
            roc_auc_macro = roc_auc_score(Y_label, Y_pred, average='macro')

            temp_df.append([str(train_patient_list), test_patient, mode, time, 'Soft Voting', ckpt, tn, fp, fn, tp,
                            str(round((sen * 100), 0)),
                            spe, ppv, str(round(fpr, 3)), acc, f1_micro, roc_auc_micro, f1_macro, roc_auc_macro])

            K.clear_session()

            df = pd.DataFrame(temp_df)
            dir_path = '.\\output'

            if not os.path.exists(dir_path):
                os.mkdir(dir_path)

            csv_path = '.\\output\\test_{test_patient}_{time}.csv'.format(test_patient=test_patient, time=time)

            df.columns = ['train_patient', 'test_patient', 'mode', 'time', 'round', 'ckpt',
                          'tn', 'fp', 'fn', 'tp', 'sen',
                          'spe', 'ppv', 'fpr', 'acc', 'f1_micro',
                          'roc_auc_micro', 'f1_macro', 'roc_auc_macro']

            df.to_csv(csv_path, index=False)

            print('================= results =================')
            cnt = 0
            total_sen = 0.
            total_fpr = 0.
            total_acc = 0.
            for i in range(len(test_result)):
                print('chb', (i + 1), test_result[i])
                total_sen += test_result[i][0]
                total_fpr += test_result[i][1]
                total_acc += test_result[i][2]
                cnt += 1
            print('Total :', (total_sen / cnt), (total_fpr / cnt), (total_acc / cnt))
        except Exception as e:
            print(traceback.format_exc())
            continue

import os

model = 'ensemble'

# Create chb list
patient = 24
chb_list = []
for i in range(patient):
    chb_list.append('chb' + str(i + 1).zfill(2))

test_files = []
train_files = []

# Length of chb list
chb_list_len = len(chb_list)

b = 0
index = 0
for chb in chb_list:
    b += 1

    try:
        time_window = 30
        test_patient = chb

        train_patient_list = []
        for kn in range(len(chb_list) - 1):
            train_patient = chb_list[(index + kn + 1) % chb_list_len]

            train_path = ".\\preprocess\\" + train_patient + "\\"

            chb_train_files = os.listdir(os.path.join(train_path, ''))
            chb_train_files.sort()
            nof = len(chb_train_files)

            if nof == 0:
                continue

            file_list = []

            for i in range(nof):
                file_list.append(chb_train_files[i])

            train_patient_list.append(train_patient)
            train_files.append(file_list)

        path = ".\\preprocess\\" + test_patient + "\\"

        files = os.listdir(os.path.join(path, ''))
        files.sort()
        nof = len(files)

        file_list = []

        for i in range(nof):
            file_list.append(files[i])

        test_files.append(file_list)

        print('train_files :', train_files)
        print('test_files :', test_files)

        lr = 1e-3
        epoch = 2
        patience = 10
        k = 5
        batch_size = 256
        train(train_patient_list=train_patient_list, test_patient=test_patient, time=time_window, mode=model, train_files=train_files, test_files=test_files, k=k, epoch=epoch, batch_size=batch_size)
        test(train_patient_list=train_patient_list, test_patient=test_patient, time=time_window, mode=model, test_files=test_files, k=k)

        train_patient_list.clear()
        train_files.clear()
        test_files.clear()

        index += 1
    except Exception as e:
        print(traceback.format_exc())
        train_files.clear()
        test_files.clear()

        index += 1