import csv
from numpy.lib.npyio import load
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, BatchNormalization, LSTM, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, RMSprop
from keras.utils.vis_utils import plot_model
import numpy as np
import pandas as pd
import os
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import random
# random.seed(1)


# =========================================================================
seed_value = 1
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.compat.v1.set_random_seed(seed_value)
# ========================GPU's utilization ========================
fraction = 0.8
os.environ["CUDA_VISIBLE_DEVICES"] = str(1)
config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = fraction
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)
# =========================================================================


latent_dim=100
optimizer = Adam(0.0002, 0.5)
traj_shape = (360, 1)      # each trajectory data shape


#-----------------------
# Load FingerPrint Data
#-----------------------

def load_data(filepath):
    fp_data = np.loadtxt(filepath, delimiter=",")
    fp_data = fp_data[:, :360]     # get difference data excluding latitude and longitude
    return fp_data



def generator(latent_dim=latent_dim):
    noise = Input(shape=(latent_dim,))
    x = Dense(64)(noise)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(128)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(360)(x)
    # x = Reshape((360, 1))(x)
    # x = LSTM(80, return_sequences=True)(x)
    gnet = Model(noise, x)
    # gnet.summary()
    return gnet


def discriminator(generated_data_shape=traj_shape):
    generated_data = Input(shape=generated_data_shape)
    # x = LSTM(80, return_sequences=True)(x)
    x = Flatten()(generated_data)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.2)(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.2)(x)
    x = Dense(128)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.2)(x)
    x = Dense(32)(x)
    x = LeakyReLU(alpha=0.2)(x)
    logit = Dense(1, activation='sigmoid')(x)
    dnet = Model(generated_data, logit)
    # print(dnet.summary())
    return dnet

# g = generator()
# d = discriminator()


def train(epochs=1000, batch_size=128, fp_data=list, latent_dim=latent_dim, generated_data_shape=traj_shape):
    losses = {"d1":[], "d2":[], "g":[], "a1":[], "a2":[]}
    
    # build discriminator
    dnet = discriminator(generated_data_shape)
    dnet.compile(loss='binary_crossentropy',
                 optimizer=optimizer,
                 metrics=['accuracy'])
    
    # the generator takes noise as input and generate trajectory
    noise = Input(shape=(latent_dim,)) 
    
    # build generator
    gnet = generator(latent_dim)


    dnet.trainable = False
    generated_data = gnet(noise)
    logit = dnet(generated_data)
    gan = Model(noise, logit)
    gan.compile(loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])

    gan.summary()
      
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    # Print Model Structure
    # plot_model(gan, to_file="./model_pict/gan_model.png", show_shapes=True)
    # plot_model(gnet, to_file="./model_pict/gnet_model.png", show_shapes=True)
    # plot_model(dnet, to_file="./model_pict/dnet_model.png", show_shapes=True)

    for epoch in range(epochs):
        print("Epoch ", epoch, ":", end=" ")
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        idx = np.random.randint(0, fp_data.shape[0], batch_size)
        fp = fp_data[idx]
        # print(fp.shape)
        
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        
        # generate a batch of new data
        generated_data = gnet.predict(noise)
        # print('generated_data: ',generated_data)
        
        # train discriminator
        d_loss_real = dnet.train_on_batch(fp, real)
        d_loss_fake = dnet.train_on_batch(generated_data, fake)
        losses['d1'].append(d_loss_real[0])
        losses['a1'].append(d_loss_real[1])
        losses['d2'].append(d_loss_fake[0])
        losses['a2'].append(d_loss_fake[1])
        
        # ---------------------
        #  Train Generator
        # ---------------------
        
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        
        # Train the generator (to have the discriminator label samples as valid)
        g_loss = gan.train_on_batch(noise, real)
        
        losses['g'].append(g_loss)
        print("d1:%2f, d2:%2f, g:%2f, a1:%2f, a2:%2f"% (d_loss_real[0], d_loss_fake[0], g_loss[0], d_loss_real[1], d_loss_fake[1]))
        # print(f'd1:{d_loss_real[0]}, d2:{d_loss_fake[0]}, g:{g_loss[0]}, a1:{d_loss_real[1]}, a2:{d_loss_fake[1]}')
        
    # with open("losses.json", "w") as outfile:
    #     json.dump(losses, outfile)

    return dnet


def load_csv(csv_files=list):
    fp_data = np.array([])
    
    for path in csv_files:
        for file in os.listdir(path):
            temp = np.array([])
            temp = load_data(path + "/" + file)
            try:
                fp_data = np.concatenate((fp_data, temp), axis=0)
            except:
                fp_data = temp
        print('fp_data', fp_data.shape)
        print(path + "/" + file + "  finished")
    
    return fp_data

if __name__ == '__main__':

    # # csv_files: folder path which store csv files
    csv_files = [
        "./Miramar_Train/Train_fixed_speed_sample",
        "./Miramar_Test/1029_fixed_speed_sample",
        "./Miramar_Test/1030_fixed_speed_sample",
        "./Miramar_Test/20201208_speed_sample",
    ]

    fp_data = load_csv(csv_files)
    # fp_data = load_data("./Miramar_Train/Suture0311.csv")
    # print(fp_data.shape)

    dnet = train(epochs=1000, batch_size=128, fp_data=fp_data, latent_dim=latent_dim,generated_data_shape=traj_shape)
    dnet.save('./Saved_Model/d_all_Miramar_0407.h5')


    # dnet = train(epochs=10000)
    # dnet.save('./Saved_Model/d_0331.h5')
