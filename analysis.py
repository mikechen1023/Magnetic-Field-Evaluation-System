from numpy.lib.npyio import load
import pandas as pd
import math
import numpy as np
import os
from keras.models import load_model
import matplotlib.pyplot as plt


modelname = "d_Miramar_Train_0402"
pict_path = "./test_analysis"


def draw_error_and_uncertainty(error, uncertainty, file):
    plt.clf()
    plt.title(f"{file}")
    plt.plot(error, uncertainty, '.')
    plt.xlabel("Distance error")
    plt.ylabel("Uncertainty")
    plt.savefig(f"./Picture/{modelname}/{file}.png")

# --------------------------------------------------------------------------------------------------------------------


def calculate_error(Predict, cor_label):
    radian = 0.0174532925
    predict_longitude = Predict[:, 1]
    predict_latitude = Predict[:, 0]
    true_longitude = cor_label[:, 1]
    true_latitude = cor_label[:, 0]
    predict_longitude = predict_longitude * radian
    predict_latitude = predict_latitude * radian
    true_longitude = true_longitude * radian
    true_latitude = true_latitude * radian
    error = []
    error = np.array(error, np.float64)
    count_domain_error = 0
    for i in range(len(Predict)):
        temp = math.sin(predict_latitude[i])*math.sin(true_latitude[i])+math.cos(true_latitude[i])*math.cos(
            predict_latitude[i])*math.cos(abs(predict_longitude[i]-true_longitude[i]))

        try:
            temp = math.acos(temp)
        except ValueError:
            error = np.append(error, 101)
            count_domain_error = count_domain_error + 1
            continue
        distance = temp*6371.009*1000
        error = np.append(error, distance)
    # print("The number of domain error is %d"%count_domain_error)
    return error

# --------------------------------------------------------------------------------------------------------------------

def error_and_uncertainty(modelpath, result_file_dir, fixed_sample_dir):
    dnet = load_model(modelpath)
    for file in os.listdir(result_file_dir):
        print(f"{file}:")
        dataset = pd.read_csv(f"{result_file_dir}/{file}")
        dataset = np.array(dataset, np.float64)
        temp0 = dataset[:, 0].reshape(-1, 1)
        temp1 = dataset[:, 1].reshape(-1, 1)
        temp2 = dataset[:, 2].reshape(-1, 1)
        temp3 = dataset[:, 3].reshape(-1, 1)
        datasetx = np.concatenate((temp0, temp1), axis=1)
        datasety = np.concatenate((temp2, temp3), axis=1)
        error = calculate_error(datasetx, datasety)
        error = np.array(error, np.float64)

        trajectory = np.loadtxt(f"{fixed_sample_dir}/{file}", delimiter=",")
        trajectory = trajectory[:, :360]
        uncertainty_list = dnet.predict(trajectory)
        uncertainty = np.array(uncertainty_list)

        draw_error_and_uncertainty(error, uncertainty, file)


def uncertainty_distr(modelpath, result_file_dir, fixed_sample_dir):

    dnet = load_model(modelpath)
    
    for file in os.listdir(result_file_dir):
        print(f"{file}:")
        trajectory = np.loadtxt(f"{fixed_sample_dir}/{file}", delimiter=",")
        trajectory = trajectory[:, :360]
        uncertainty_list = dnet.predict(trajectory)
        uncertainty = np.array(uncertainty_list)
        bins_list = [(i/10) for i in range(0, 10)]
        high = [i for i in uncertainty if i >= 0.8 ]
        med = [i for i in uncertainty if i>0.6 and i<0.8]
        low = [i for i in uncertainty if i<=0.6]
        plt.clf()
        plt.title(file)
        plt.xlabel("range")
        plt.ylabel("count")
        plt.hist(uncertainty, bins=bins_list, label = f"total:{len(uncertainty)}\nhigh:{len(high)}\nmed:{len(med)}\nlow:{len(low)}")
        plt.legend()
        plt.savefig(f"./Picture/Distribution/{modelname}/{file}.png")
    


if __name__ == '__main__':

    
    modelpath = f"./Saved_Model/{modelname}.h5"

    result_file_dir = "./Test/1029_Miramar_result"
    fixed_sample_dir = "./Test/1029_fixed_speed_sample"
    # result_file_dir = "./Test/Test_Outdomain_data_result"
    # fixed_sample_dir = "./Test/Test_Outdomain_data"
    error_and_uncertainty(modelpath, result_file_dir, fixed_sample_dir)
    uncertainty_distr(modelpath, result_file_dir, fixed_sample_dir)