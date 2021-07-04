# %%

import scipy.io as sio
import scipy.misc as smisc
import numpy as np

# %%

import h5py
from tqdm import tqdm
import random
import threading
import os
import sys
import math
import time
import pickle
from PIL import Image
import io
import gc
import psutil


# new imports for pytorch port

import torch

import tensorflow as tf

tf.compat.v1.disable_eager_execution()
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Lambda, Activation, \
    Dropout, Flatten
from tensorflow.keras.layers import concatenate, add, Layer
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import backend as K


from weight_dropout import Conv2D_wd
from gaussian_logit_sampling import GaussianLogitSampler
from dataloader import DataLoader

# %%

def mae_plus_derv_cvae_loss(true, pred):
    true_x_1 = K.concatenate([true[:, 1:, :, :], K.zeros_like(true[:, 0:1, :, :])], axis=1)
    true_y_1 = K.concatenate([true[:, :, 1:, :], K.zeros_like(true[:, :, 0:1, :])], axis=2)
    pred_x_1 = K.concatenate([pred[:, 1:, :, :], K.zeros_like(pred[:, 0:1, :, :])], axis=1)
    pred_y_1 = K.concatenate([pred[:, :, 1:, :], K.zeros_like(pred[:, :, 0:1, :])], axis=2)

    diff_x_true = K.abs(true - true_x_1)
    diff_y_true = K.abs(true - true_y_1)
    diff_x_pred = K.abs(pred - pred_x_1)
    diff_y_pred = K.abs(pred - pred_y_1)

    diff_der = K.abs(diff_x_true - diff_x_pred) + K.abs(diff_y_true - diff_y_pred)
    return 1.0 * K.mean(K.abs(pred - true), axis=-1) + K.mean(diff_der, axis=-1)


# %%

def res_block(input_tensor, res_func, num_of_output_feature_maps):
    input_tensor = Conv2D(num_of_output_feature_maps, (1, 1))(input_tensor)
    res_func = add([input_tensor, res_func])
    res_func = Activation('relu')(res_func)
    return res_func


# %%

def get_model_disc_im(input_shape):
    input1 = Input(shape=input_shape)

    initializer = 'glorot_uniform'

    decoder = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer=initializer)(input1)
    decoder = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer=initializer)(decoder)
    decoder = MaxPooling2D((2, 2))(decoder)

    decoder = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer=initializer)(decoder)
    decoder = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer=initializer)(decoder)
    decoder = MaxPooling2D((2, 2))(decoder)

    decoder = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer=initializer)(decoder)
    decoder = MaxPooling2D((2, 2))(decoder)
    decoder = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer=initializer)(decoder)
    decoder = MaxPooling2D((2, 2))(decoder)

    decoder = Flatten()(decoder)
    decoder = Dense(1024, activation='relu', kernel_initializer=initializer)(decoder)
    decoder = Dense(256, activation='relu', kernel_initializer=initializer)(decoder)

    decoder = Dense(2, activation='softmax', kernel_initializer=initializer)(decoder)

    model = Model(inputs=[input1], outputs=decoder)

    model.compile(optimizer=Adam(1.0 * 1e-4), loss='categorical_crossentropy')

    return model


# %%

def get_model(input_shape1, input_shape2):
    image_input = Input(shape=input_shape1)

    odo_input = Input(shape=input_shape2)

    initializer = 'glorot_uniform'

    odo_enc = TimeDistributed(Dense(8, activation='relu', kernel_initializer=initializer))(odo_input)
    odo_enc = Reshape((5 * 8,))(odo_enc)
    odo_enc = Dense(32, activation='relu', kernel_initializer=initializer)(odo_enc)
    odo_enc = Dense(8, activation='relu', kernel_initializer=initializer)(odo_enc)
    odo_enc = Reshape((1, 1, 8))(odo_enc)
    odo_enc = UpSampling2D(size=(128, 256))(odo_enc)

    encoder1_1_ = concatenate([image_input, odo_enc], axis=-1)
    encoder1_1_ = Conv2D(64 * 2, (3, 3), padding='same', activation='relu', kernel_initializer=initializer)(encoder1_1_)
    #     encoder1_3 = Conv2D(64*2,( 3, 3), padding='same', activation='relu',kernel_initializer=initializer)(encoder1_1_)
    #     encoder1_3 = Dropout(.2)(encoder1_3)
    #     encoder1_5 = Conv2D(64*2,( 3, 3), padding='same', activation='relu',kernel_initializer=initializer)(encoder1_3)
    #     encoder1_5 = Dropout(.2)(encoder1_5)
    encoder1_3 = Conv2D_wd(64 * 2, (3, 3), padding='same', activation='relu', drop_prob=0.20)(encoder1_1_)
    encoder1_5 = Conv2D_wd(64 * 2, (3, 3), padding='same', activation='relu', drop_prob=0.20)(encoder1_3)
    encoder1_6 = add([encoder1_1_, encoder1_5])
    encoder1_7 = Activation('relu')(encoder1_6)
    encoder2 = MaxPooling2D(pool_size=(2, 2))(encoder1_7)

    #     encoder3_1 = Conv2D(128*2,( 3, 3), padding='same', activation='relu',kernel_initializer=initializer)(encoder2)
    #     encoder3_1 = Dropout(.2)(encoder3_1)
    #     encoder3_3 = Conv2D(128*2,( 3, 3), padding='same', activation='relu',kernel_initializer=initializer)(encoder3_1)
    #     encoder3_3 = Dropout(.2)(encoder3_3)
    #     encoder3_5 = Conv2D(128*2,( 3, 3), padding='same', activation='relu',kernel_initializer=initializer)(encoder3_3)
    #     encoder3_5 = Dropout(.2)(encoder3_5)
    encoder3_1 = Conv2D_wd(128 * 2, (3, 3), padding='same', activation='relu', drop_prob=0.20)(encoder2)
    encoder3_3 = Conv2D_wd(128 * 2, (3, 3), padding='same', activation='relu', drop_prob=0.20)(encoder3_1)
    encoder3_5 = Conv2D_wd(128 * 2, (3, 3), padding='same', activation='relu', drop_prob=0.20)(encoder3_3)
    encoder3_7 = res_block(encoder2, encoder3_5, 128 * 2)
    encoder4 = MaxPooling2D(pool_size=(2, 2))(encoder3_7)

    #     encoder5_1 = Conv2D(256*2,( 3, 3), padding='same', activation='relu',kernel_initializer=initializer)(encoder4)
    #     encoder5_1 = Dropout(.2)(encoder5_1)
    #     encoder5_3 = Conv2D(256*2,( 3, 3), padding='same', activation='relu',kernel_initializer=initializer)(encoder5_1)
    #     encoder5_3 = Dropout(.2)(encoder5_3)
    #     encoder5_5 = Conv2D(256*2,( 3, 3), padding='same', activation='relu',kernel_initializer=initializer)(encoder5_3)
    #     encoder5_5 = Dropout(.2)(encoder5_5)
    encoder5_1 = Conv2D_wd(256 * 2, (3, 3), padding='same', activation='relu', drop_prob=0.20)(encoder4)
    encoder5_3 = Conv2D_wd(256 * 2, (3, 3), padding='same', activation='relu', drop_prob=0.20)(encoder5_1)
    encoder5_5 = Conv2D_wd(256 * 2, (3, 3), padding='same', activation='relu', drop_prob=0.20)(encoder5_3)
    encoder5_7 = res_block(encoder4, encoder5_5, 256 * 2)
    encoder6 = UpSampling2D(size=(2, 2))(encoder5_7)

    #     encoder7_1 = Conv2D(128*2,( 3, 3), padding='same', activation='relu',kernel_initializer=initializer)(encoder6)
    #     encoder7_1 = Dropout(.2)(encoder7_1)
    #     encoder7_3 = Conv2D(128*2,( 3, 3), padding='same', activation='relu',kernel_initializer=initializer)(encoder7_1)
    #     encoder7_3 = Dropout(.2)(encoder7_3)
    #     encoder7_5 = Conv2D(128*2,( 3, 3), padding='same', activation='relu',kernel_initializer=initializer)(encoder7_3)
    #     encoder7_5 = Dropout(.2)(encoder7_5)
    encoder7_1 = Conv2D_wd(128 * 2, (3, 3), padding='same', activation='relu', drop_prob=0.20)(encoder6)
    encoder7_3 = Conv2D_wd(128 * 2, (3, 3), padding='same', activation='relu', drop_prob=0.20)(encoder7_1)
    encoder7_5 = Conv2D_wd(128 * 2, (3, 3), padding='same', activation='relu', drop_prob=0.20)(encoder7_3)
    encoder7_7 = res_block(encoder6, encoder7_5, 128 * 2)
    encoder8 = UpSampling2D(size=(2, 2))(encoder7_7)

    #     encoder9_1 = Conv2D(64*2,( 3, 3), padding='same', activation='relu', kernel_initializer=initializer)(encoder8)
    #     encoder9_1 = Dropout(.2)(encoder9_1)
    #     encoder9_3 = Conv2D(64,( 3, 3), padding='same', activation='relu', kernel_initializer=initializer)(encoder8)
    #     encoder9_3 = Dropout(.2)(encoder9_3)
    encoder9_1 = Conv2D_wd(64 * 2, (3, 3), padding='same', activation='relu', drop_prob=0.20)(encoder8)
    encoder9_3 = Conv2D_wd(64, (3, 3), padding='same', activation='relu', drop_prob=0.20)(encoder9_1)

    encoder9_5 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer=initializer)(encoder9_3)

    encoder = Conv2D(24 * 2, (3, 3), padding='same', kernel_initializer=initializer)(encoder9_5)  # used to be 19*2

    encoder = GaussianLogitSampler()(encoder)

    full_model = Model(inputs=[image_input, odo_input], outputs=encoder)

    return full_model


# %%

def get_IOU(opred, otrue):
    pred = tf.compat.v1.placeholder(tf.int32, shape=(1, 128, 256, 1))
    label = tf.compat.v1.placeholder(tf.int32, shape=(1, 128, 256, 1))

    label = tf.compat.v1.Print(label, [label], message="This is a: ")

    pred_flatten = tf.reshape(pred, [-1, ])
    raw_gt = tf.reshape(label, [-1, ])

    indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, 23)), 1)
    gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
    fpred = tf.gather(pred_flatten, indices)

    miou = tf.math.confusion_matrix(fpred, gt, num_classes=24)
    sess = tf.compat.v1.Session()
    init1 = tf.compat.v1.global_variables_initializer()
    init2 = tf.compat.v1.local_variables_initializer()
    sess.run(init1)
    sess.run(init2)

    _miou = 0.0;
    gconf_m = np.zeros((24, 24));
    for i in range(0, opred.shape[0], 1):
        _pred = np.expand_dims(opred[i:i + 1].argmax(axis=3), axis=3).astype(int)
        _label = otrue[i:i + 1].astype(int)
        _miou = sess.run(miou, feed_dict={pred: _pred, label: _label})
        gconf_m += _miou

    I = np.diag(np.diag(gconf_m))
    U = np.diag(np.sum(gconf_m - I, axis=0) + np.sum(gconf_m - I, axis=1)) + I + 0.00000000001
    # print((np.diag(I / U)))
    return np.mean(np.diag(I / U))


# %%

def most_accurate(_preds, data_Y, test_samples):
    acc_mat = np.zeros((_preds.shape[0], test_samples));
    for i in range(0, _preds.shape[0]):
        for sample in range(0, test_samples):
            _pred = _preds[i, sample, :, :, :].argmax(axis=2)
            _pred = np.expand_dims(_pred, axis=2).astype(int)
            _label = data_Y[i, :, :, :].astype(int)
            acc_mat[i] = float(np.sum((_pred == _label).astype(int))) / (256 * 128)

    best_preds = [];
    for i in range(0, _preds.shape[0]):
        best_preds.append(np.mean(_preds[i, np.argsort(acc_mat[i, :])[-3:], :, :, :], axis=0))

    _data_Y = data_Y.argmax(axis=-1).astype(int)
    _data_Y = np.expand_dims(_data_Y, axis=3).astype(int)
    _best_preds = np.array(best_preds)
    _best_preds = _best_preds.argmax(axis=-1).astype(int)
    _best_preds = np.expand_dims(_best_preds, axis=3).astype(int)
    return _best_preds, _data_Y


# %%

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val


# %%

discriminator_im = get_model_disc_im((128, 256, 24 * 5))
discriminator_im.summary()

# %%

full_model = get_model((128, 256, 24 * 4), (5, 2));
full_model.summary()

# %%

gan_input_im = Input(shape=(128, 256, 24 * 4))
gan_input_odo = Input(shape=(5, 2))
gan_input_cond = Input(shape=(128, 256, 24 * 4))
H = full_model([gan_input_im, gan_input_odo])
print(H.shape)

# %%

H_wc = concatenate([gan_input_cond, H])
gan_dis = discriminator_im(H_wc)

# %%

make_trainable(discriminator_im, False)
GAN = Model(inputs=[gan_input_im, gan_input_odo, gan_input_cond], outputs=[H, gan_dis])

GAN.compile(optimizer=Adam(1e-4), loss=[mae_plus_derv_cvae_loss, 'categorical_crossentropy'], loss_weights=[1.0, 0.01])

GAN.summary()

print('Done.')

# %%

batch_size = 8;

dataloader = DataLoader(batch_size)

print('Loading val data ...')

test_examples = 1

test_samples = 5

(data_val_X_f, data_val_X_o, data_val_Y, data_val_Y_) = dataloader.get_val_data(test_examples,
                                                                                test_samples)  # , data_X_o

print('Done.')

# print (psutil.virtual_memory())
# print (psutil.swap_memory())

# %%

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=Adam(1e-4),
                                 discriminator_optimizer=Adam(1e-4),
                                 generator=GAN,
                                 discriminator=discriminator_im
                                 )
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# %%

# l_loss_d = [];
# l_loss_g = [];
epochs = 50
for _ in range(epochs):
    steps = 4000
    start = time.time()
    for step in tqdm(range(1, steps + 1)):
        # -----------------------------------------------------------------------
        # Train Discriminator for Synthetic Likelihood
        # -----------------------------------------------------------------------
        data_X_s_batch, data_X_o_batch, data_Y_batch = dataloader.train_data_batch()
        generated_y = full_model.predict([data_X_s_batch, data_X_o_batch], batch_size=batch_size)

        data_Y_batch_wc = np.concatenate([data_X_s_batch, data_Y_batch], axis=-1)
        generated_y_wc = np.concatenate([data_X_s_batch, generated_y], axis=-1)

        X = np.concatenate((data_Y_batch_wc, generated_y_wc))  # + np.random.normal(0,0.05,size=y_batch.shape)
        y = np.zeros([2 * batch_size, 2]) + np.random.uniform(0, 0.05, size=(2 * batch_size, 2))
        y[0:batch_size, 1] = 1 - np.random.uniform(0, 0.05, size=(batch_size,))
        y[batch_size:, 0] = 1 - np.random.uniform(0, 0.05, size=(batch_size,))

        make_trainable(discriminator_im, True)
        d_loss = discriminator_im.train_on_batch([X], y)

        #         l_loss_d.append(d_loss)
        #         if len(l_loss_d) > 1000：
        #            l_loss_d.pop(0)

        # -----------------------------------------------------------------------
        # Train Generator
        # -----------------------------------------------------------------------

        data_X_s_batch, data_X_o_batch, data_Y_batch = dataloader.train_data_batch()

        make_trainable(discriminator_im, False)

        y2 = np.zeros([batch_size, 2])
        y2[:, 1] = 1 - np.random.uniform(0, 0.05, size=(batch_size,))

        g_loss = GAN.train_on_batch([data_X_s_batch, data_X_o_batch, data_X_s_batch], [data_Y_batch, y2])

        #         l_loss_g.append(g_loss);
        #         if len(l_loss_g) > 1000:
        #             l_loss_g.pop(0)

        if step % 10 == 0:
            tqdm.write('Step: ' + str(step) + ' -- Loss - D: ' + str(d_loss) + ' -- Loss - G: ' + str(g_loss))

        if step % 2000 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        if step % 50 == 0:
            del data_X_s_batch, data_X_o_batch, data_Y_batch, generated_y, data_Y_batch_wc, generated_y_wc, X, y, y2

            gc.collect()

    print('Validating ...')

    K.set_learning_phase(0);

    preds = full_model.predict([data_val_X_f[0:data_val_X_f.shape[0], ], data_val_X_o[0:data_val_X_f.shape[0], ]],
                               verbose=1, batch_size=1);

    index = int(preds.shape[0] / test_samples)

    preds = np.reshape(preds, (index, test_samples, preds.shape[1], preds.shape[2], preds.shape[3]))

    preds, val_y = most_accurate(preds, data_val_Y, test_samples);

    miou = get_IOU(preds, val_y);

    print('mIoU: ', miou)

    elapsed = time.time() - start

    print('This epoch took: ' + str(float(elapsed) / (60 * 60)) + ' hours')

    start = time.time();

    print('Continuing training ...')