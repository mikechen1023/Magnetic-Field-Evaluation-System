import numpy
import numpy as np
import matplotlib.pyplot as plt
import csv
import numpy
import pandas as pd
import os
import random
import math
import statsmodels.api as sm
import seaborn as sns
import dc_stat_think as dcst
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, default="Mirmar_cdf95")
args = parser.parse_args()


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
# ---------------------------------------------------------------------------------------------------------------------------------------


def CDF(x, y, num):
    val = np.percentile(x, [num])
    if num == 95 or num==68:
        print("CEP%d is about %.3f" % (num, val))
    return val
    # print("")

# ---------------------------------------------------------------------------------------------------------------------------------------


def route(file_dir, uncertainty_dir, cdf_dir, date):
    test_df = pd.DataFrame([], columns=['Data', 'CDF68', 'CDF95'])
    for file in os.listdir(file_dir):
        print(f"{file}:")
        row_dict = {}
        df = pd.read_csv(f"{file_dir}/{file}")
        dataset = df.to_numpy()
        dataset = np.array(dataset, np.float64)
        temp0 = dataset[:, 0].reshape(-1, 1)
        temp1 = dataset[:, 1].reshape(-1, 1)
        temp2 = dataset[:, 2].reshape(-1, 1)
        temp3 = dataset[:, 3].reshape(-1, 1)
        datasetx = np.concatenate((temp0, temp1), axis=1)
        datasety = np.concatenate((temp2, temp3), axis=1)
        error = calculate_error(datasetx, datasety)
        error = np.array(error, np.float64)
        

        plt.clf()
        plt.title(f"{file}")
        plt.plot(error, dataset[:, 4], '.')
        plt.plot(error[dataset[:,4]<2],dataset[dataset[:,4]<2,4],'.r')
        #corr = np.corrcoef(error, dataset[:, 4])
        #corr = corr[0,1]
        # sum = 0
        # for i in range(len(error)):
        #     if dataset[i, 4] < 4:
        #         sum = sum+1
        #         plt.plot(error[i], dataset[i, 4], '.', color="r")
        # print(sum/len(error))
        plt.xlabel('Distance error')
        plt.ylabel('Uncertainty')
        #plt.annotate(f'{corr:.02f}',xy = (1,1))
        plt.savefig(f"{uncertainty_dir}/{file}.png")



        plt.clf()
        #plt.plot(dataset[:,1],dataset[:,0],'o',color='b')
        plt.plot(dataset[:,3],dataset[:,2],'o',color='black',markersize = 1)
        plt.scatter(dataset[:,3],dataset[:,2],s =dataset[:,4]*20,alpha = 0.7)
        plt.plot(dataset[error>5,3],dataset[error>5,2],'o',color='red',markersize = 3)
        plt.title(f"{file}")    
        #plt.xticks(fontsize=2)
        #plt.yticks(fontsize=2)
        uncertainty_dir1 = "./Picture/Test"
        plt.savefig(f"{uncertainty_dir1}/{file}.png")


        x, y = dcst.ecdf(error)
        CDF(x, y, 68)
        plt.clf()
        cdf95 = CDF(x, y, 95)
        cdf68 = CDF(x, y, 68)
        cdf = [CDF(x, y, i) for i in range(101)]
        plt.title(f"CDF {file}")
        plt.plot(cdf,[i for i in range(101)])
        plt.savefig(f"cdf95/{file}.png")
        row_dict["Data"] = file
        row_dict["CDF68"] = cdf68
        row_dict["CDF95"] = cdf95
        test_df = pd.concat([test_df, pd.DataFrame.from_dict(
            row_dict, orient='columns')], ignore_index=True)
        print("")
    test_df.to_csv(f"{cdf_dir}/{args.output}_{date}.csv", index=False)


uncertainty_dir = "./Picture/Uncertainty"
cdf_dir = "cdf95"

date = "1029"
file_dir = "1029_Miramar_result"
route(file_dir, uncertainty_dir, cdf_dir, date)

date = "1030"
file_dir = "1030_Miramar_result"
route(file_dir, uncertainty_dir, cdf_dir, date)

date = "1208"
file_dir = "1208_Miramar_result"
route(file_dir, uncertainty_dir, cdf_dir, date)
