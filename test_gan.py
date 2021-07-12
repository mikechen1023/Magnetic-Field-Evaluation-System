import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

# in-domain
filename = "20201029_Miramar_Final_MagMap_NCTU_new.csv"
filepath = "./Train_fixed_speed_sample/"+filename

# out-domain
# filename = "Final_MagMap_outage_20210114_Taipei101_mapping_B2_vertical_middle.csv"
# filepath = "../Test_Outdomain_data/"+filename

modelname = "d_Miramar_Train_0402"
modelpath = "./Saved_Model/" + modelname + ".h5"




# plt.scatter(dataset[:,3], dataset[:,2], s=dataset[:,4]*20, alpha = 0.7)
def draw_uncertainty(dataset, output, title, save_path=None):
    fig = plt.figure()
    plt.plot(dataset[:, 1], dataset[:, 0], 'o', color='black', markersize=1)
    plt.scatter(dataset[:, 1], dataset[:, 0], s=50*output, alpha=0.7)
    plt.title(title)
    if save_path is None:
        plt.savefig(f"./test_0405/{modelname}_out-domain.png")
    plt.show()

# load discriminator model
dnet = load_model(modelpath)
# print(dnet.summary())

# load test data
test_data = np.loadtxt(filepath, delimiter=",")
test_data_loc = test_data[:, -2:] # load longitude and latitude
test_data = test_data[:, :360]
print(test_data_loc)
# print(test_data.shape)

test_output = dnet.predict(test_data)
print(test_output.dtype)
# print(test_output, sep="\n")
draw_uncertainty(test_data_loc, test_output, filename)

high = 0
med = 0
low = 0

for logit in test_output:
    if logit >= 0.9:
        high += 1
    elif logit >=0.6 and logit < 0.9:
        med += 1
    else:
        low += 1

print("\ntest data number:", len(test_output))
print("high: ",high)
print("med: ", med)
print("low: ", low)


    
