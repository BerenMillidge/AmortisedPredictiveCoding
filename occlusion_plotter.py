import numpy as np
import matplotlib.pyplot as plt


variational_acclist = np.load("proper_occlusion_num_occluder_variational_acclist.npy")
amortised_acclist = np.load("proper_occlusion_num_occluder_amortised_acclist.npy")
ANN_acclist = np.load("ANN_n_occluder_acc_list.npy")
variational_only_acclist = np.load("variational_only_num_occluder_variational_acclist.npy")


print(variational_acclist)
print(amortised_acclist)
print(ANN_acclist)
print(variational_only_acclist)

fig = plt.figure()
xs = np.array([1,3,5,7,9])
plt.title("Performance vs Number of Occluders")
plt.bar(xs-0.25,variational_acclist,width=0.25,label="Hybrid Variational")
plt.bar(xs,amortised_acclist,width=0.25,label="Hybrid Amortised")
plt.bar(xs+0.25,ANN_acclist,width=0.25,label="ANN")
plt.bar(xs + 0.5, variational_only_acclist,width=0.25, label="Variational Only")
plt.xlabel("Number of Occluders")
plt.ylabel("Test Accuracy")
plt.legend()
plt.show()

variational_acclist = np.load("proper_occlusion_occluder_size_variational_acclist.npy")
amortised_acclist = np.load("proper_occlusion_occluder_size_amortised_acclist.npy")
ANN_acclist = np.load("ANN_occluder_size_acc_list.npy")
variational_only_acclist = np.load("variational_only_occluder_size_variational_acclist.npy")


print(variational_acclist)
print(amortised_acclist)
print(ANN_acclist)
print(variational_only_acclist)

fig = plt.figure()
xs = np.array([1,2,3,4,5,6,7])
plt.title("Performance vs Occluder Sizes")
plt.bar(xs-0.25,variational_acclist,width=0.25,label="Hybrid Variational")
plt.bar(xs,amortised_acclist,width=0.25,label="Hybrid Amortised")
plt.bar(xs+0.25,ANN_acclist,width=0.25,label="ANN")
plt.bar(xs+0.5,variational_only_acclist,width=0.25,label="Variational Only")
plt.xlabel("Occluder Size (NxN)")
plt.ylabel("Test Accuracy")
plt.legend()
plt.show()

hybrid_var_mean = np.mean(variational_acclist)
hybird_amort_mean = np.mean(amortised_acclist)
ANN_mean = np.mean(ANN_acclist)
variational_only_mean = np.mean(variational_only_acclist)
data = [hybrid_var_mean,hybird_amort_mean,ANN_mean,variational_only_mean]
fig = plt.figure()
plt.title("Average Performance")
xs = ["Hybrid Variational","Hybrid Amortised","ANN","Variational Only"]
plt.bar(xs,data)
plt.ylabel("Average Test Accuracy")
plt.xticks(fontsize=9)
plt.show()
