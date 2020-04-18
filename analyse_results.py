import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
import os
import sys
basepath = "edinburgh_logs/edinburgh_logs/edinburgh_logs/"
old_basepath = "edinburgh_experiments/edinburgh_experiments/edinburgh_experiments/"
plt.style.use(['seaborn-white', 'seaborn-paper'])
matplotlib.rc("font", family="Times New Roman")
FONT_SIZE = 20
plt.rc('ytick', labelsize=16)
plt.rc('xtick', labelsize=16)
plt.rc('legend', fontsize=FONT_SIZE)

plt.rcParams['axes.edgecolor'] = '#333F4B'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.color'] = '#333F4B'
plt.rcParams['ytick.color'] = '#333F4B'

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'

def plot_error_dispersion_connections(standard_sname ="standard",error_dispersion_sname="error_dispersion_with_weight_updates",no_error_update_sname="proper_error_dispersion_without_weight_updates"):
    standard_path = basepath + standard_sname
    error_dispersion_path = basepath + error_dispersion_sname
    no_weight_updates_path = basepath + no_error_update_sname
    dirs = os.listdir(basepath)
    standard_accs = []
    error_accs = []
    no_update_accs = []
    for i in range(5):
        spath = standard_path +"/"+str(i)
        errpath = error_dispersion_path + "/" + str(i)
        nweightpath = no_weight_updates_path + "/" + str(i)
        standard_accs.append(np.load(spath+"/variational_acc.npy"))
        error_accs.append(np.load(errpath+"/variational_acc.npy"))
        no_update_accs.append(np.load(nweightpath+"/variational_acc.npy"))

    xs = np.linspace(0,101,101)
    ys = np.linspace(0,101,21)
    print("NO UPDATES: ", no_update_accs)
    print("NO UPDATE ACCS: ", len(no_update_accs))
    standard_accs = np.array(standard_accs)
    #print("standard accs: ", standard_accs)
    error_accs = np.array(error_accs)
    no_update_accs = np.array(no_update_accs)
    standard_means = np.mean(standard_accs, axis=0)
    error_means = np.mean(error_accs, axis=0)
    no_update_means = np.mean(no_update_accs, axis=0)
    standard_stds = np.std(standard_accs,axis=0)
    error_stds = np.std(error_accs,axis=0)
    no_update_stds = np.std(no_update_accs,axis=0)
    #begin plotting
    f, ax = plt.subplots(1, 1, figsize=(10, 7))
    plt.style.use(['seaborn-white', 'seaborn-paper'])
    plt.rcParams['axes.edgecolor'] = '#333F4B'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.color'] = '#333F4B'
    plt.rcParams['ytick.color'] = '#333F4B'

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
    ax.fill_between(xs, standard_means - standard_stds,standard_means + standard_stds, alpha=0.1,color='#000000')
    ax.plot(xs, standard_means, label='Standard Predictive Coding',alpha=0.63,color='black')
    ax.fill_between(xs, error_means - error_stds, error_means + error_stds, alpha=0.1,color='#8D0101')
    ax.plot(xs, error_means, label="Error connections",color='#8D0101')
    ax.fill_between(xs, no_update_means - no_update_stds, no_update_means + no_update_stds, alpha=0.1,color='#EDC31B')
    ax.plot(xs, no_update_means, label="No updates",color='#EDC31B')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['left'].set_smart_bounds(True)
    #ax.spines['bottom'].set_smart_bounds(True)
    plt.xlabel("Epoch",fontsize=12,labelpad=17,style='oblique')
    plt.ylabel("Test Accuracy",fontsize=15,labelpad=20,style='oblique')
    plt.title("Error Dispersion Accuracy",fontsize=16,fontweight='bold',pad=25)
    leg = plt.legend(fontsize=12)
    frame = leg.get_frame()
    frame.set_facecolor('1.0')
    frame.set_edgecolor('1.0')
    plt.rc('ytick', labelsize=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('legend', fontsize=FONT_SIZE)
    xticks = np.arange(0,100,10)
    plt.xticks(xticks)
    plt.tight_layout()
    f.savefig("error_connections.png")
    plt.show()

def plot_nonlinearities(standard_sname ="standard",nonlinearities_sname="no_backward_nonlinearities"):
    standard_path = basepath + standard_sname
    no_nonlinearities_path = basepath + nonlinearities_sname
    dirs = os.listdir(basepath)
    standard_accs = []
    nonlinearities_accs = []
    for i in range(5):
        spath = standard_path +"/"+str(i)
        nonlinearitiespath = no_nonlinearities_path + "/" + str(i)
        standard_accs.append(np.load(spath+"/variational_acc.npy"))
        nonlinearities_accs.append(np.load(nonlinearitiespath+"/variational_acc.npy"))

    xs = np.linspace(0,101,101)
    ys = np.linspace(0,101,21)
    standard_accs = np.array(standard_accs)
    nonlinearities_accs =np.array(nonlinearities_accs)
    standard_means = np.mean(standard_accs, axis=0)
    nonlinearities_means = np.mean(nonlinearities_accs,axis=0)
    standard_stds = np.std(standard_accs,axis=0)
    nonlinearities_std = np.std(nonlinearities_accs,axis=0)
    #begin plotting
    f, ax = plt.subplots(1, 1, figsize=(10, 7))
    plt.style.use(['seaborn-white', 'seaborn-paper'])
    plt.rcParams['axes.edgecolor'] = '#333F4B'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.color'] = '#333F4B'
    plt.rcParams['ytick.color'] = '#333F4B'

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
    ax.fill_between(xs, standard_means - standard_stds,standard_means + standard_stds, alpha=0.1,color='#000000')
    ax.plot(xs, standard_means, label='Standard Predictive Coding',color='black')
    ax.fill_between(xs, nonlinearities_means - nonlinearities_std, nonlinearities_means + nonlinearities_std, alpha=0.1,color='#8D0101')
    ax.plot(xs, nonlinearities_means, label="No Backward Nonlinearities",color='#8D0101')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['left'].set_smart_bounds(True)
    #ax.spines['bottom'].set_smart_bounds(True)
    plt.xlabel("Epoch",fontsize=12,labelpad=17,style='oblique')
    plt.ylabel("Test Accuracy",fontsize=15,labelpad=20,style='oblique')
    plt.title("No Backward Nonlinearities Accuracy",fontsize=16,fontweight='bold',pad=25)
    leg = plt.legend(fontsize=12)
    frame = leg.get_frame()
    frame.set_facecolor('1.0')
    frame.set_edgecolor('1.0')
    plt.rc('ytick', labelsize=12)
    plt.rc('xtick', labelsize=12)
    plt.rc('legend', fontsize=FONT_SIZE)
    xticks = np.arange(0,100,10)
    plt.xticks(xticks)
    #plt.tight_layout()
    f.savefig("no_nonlinearities.png")
    plt.show()


def plot_backwards_weights(standard_sname ="standard",backward_weight_sname="backward_weight_update",no_update_sname="no_backward_weight_update"):
    standard_path = basepath + standard_sname
    backward_weight_path = basepath + backward_weight_sname
    no_weight_updates_path = basepath + no_update_sname
    dirs = os.listdir(basepath)
    standard_accs = []
    backward_weight_accs = []
    no_update_accs = []
    for i in range(5):
        spath = standard_path +"/"+str(i)
        bweightpath = backward_weight_path + "/" + str(i)
        nweightpath = no_weight_updates_path + "/" + str(i)
        standard_accs.append(np.load(spath+"/variational_acc.npy"))
        backward_weight_accs.append(np.load(bweightpath+"/variational_acc.npy"))
        no_update_accs.append(np.load(nweightpath+"/variational_acc.npy"))

    xs = np.linspace(0,101,101)
    ys = np.linspace(0,101,21)
    standard_accs = np.array(standard_accs)
    #print("standard accs: ", standard_accs)
    backward_weight_accs = np.array(backward_weight_accs)
    no_update_accs = np.array(no_update_accs)
    standard_means = np.mean(standard_accs, axis=0)
    backward_weight_means = np.mean(backward_weight_accs, axis=0)
    no_update_means = np.mean(no_update_accs, axis=0)
    standard_stds = np.std(standard_accs,axis=0)
    backward_weight_stds = np.std(backward_weight_accs,axis=0)
    no_update_stds = np.std(no_update_accs,axis=0)
    #begin plotting
    f, ax = plt.subplots(1, 1, figsize=(10, 7))
    plt.style.use(['seaborn-white', 'seaborn-paper'])
    plt.rcParams['axes.edgecolor'] = '#333F4B'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.color'] = '#333F4B'
    plt.rcParams['ytick.color'] = '#333F4B'

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
    ax.fill_between(xs, standard_means - standard_stds,standard_means + standard_stds, alpha=0.1,color='black')
    ax.plot(xs, standard_means, label='Standard Predictive Coding', alpha=0.63, color='#000000')
    ax.fill_between(xs, backward_weight_means - backward_weight_stds, backward_weight_means + backward_weight_stds, alpha=0.1,color='#8D0101')
    ax.plot(xs, backward_weight_means, label="Backward Weights",color='#8D0101')
    ax.fill_between(xs, no_update_means - no_update_stds, no_update_means + no_update_stds, alpha=0.1,color='#EDC31B')
    ax.plot(xs, no_update_means, label="No updates",color='#EDC31B')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['left'].set_smart_bounds(True)
    #ax.spines['bottom'].set_smart_bounds(True)
    plt.xlabel("Epoch",fontsize=12,labelpad=17,style='oblique')
    plt.ylabel("Test Accuracy",fontsize=15,labelpad=20,style='oblique')
    plt.title("Backward Weight Accuracy",fontsize=16,fontweight='bold',pad=25)
    leg = plt.legend(fontsize=12)
    frame = leg.get_frame()
    frame.set_facecolor('1.0')
    frame.set_edgecolor('1.0')
    plt.rc('ytick', labelsize=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('legend', fontsize=FONT_SIZE)
    xticks = np.arange(0,100,10)
    plt.xticks(xticks)
    plt.tight_layout()
    f.savefig("backwards_weights.png")
    plt.show()

def plot_weight_diffs(sname="backward_weight_diffs"):
    standard_path = basepath + sname
    dirs = os.listdir(basepath)
    standard_accs = [[] for i in range(5)]
    for i in range(5):
        for l in range(3):
            spath = standard_path +"/"+str(i)+"/weight_differences_layer_"+str(l)+".npy"
            standard_accs[i].append(np.load(spath))
        standard_accs[i] = np.array(standard_accs[i])
    standard_accs = np.array(standard_accs) #Seed X Layers X N tensor
    mean_accs = np.mean(standard_accs,axis=0)
    std_accs = np.std(standard_accs,axis=0)
    for l in range(3):
        plt.plot(mean_accs[l,:])
        plt.show()
    print(standard_accs[0,0,:])




if __name__=='__main__':
    plot_error_dispersion_connections()
    #plot_nonlinearities()
    #plot_backwards_weights()
    #plot_weight_diffs()





"""sname = "no_backward_nonlinearities"
path = basepath + sname
dirs = os.listdir(path)
var_accs = []
am_accs = []
test_var_accs = []
test_am_accs = []
for i in range(len(dirs)):
    p = path + "/" + str(dirs[i])
    print("P: ",p)
    var_accs.append(np.load(p+"/variational_acc.npy"))
    am_accs.append(np.load(p+"/amortised_acc.npy"))
    test_var_accs.append(np.load(p+"/test_variational_acc.npy"))
    test_am_accs.append(np.load(p+"/test_amortised_acc.npy"))

var_accs = np.array(var_accs)
am_accs = np.array(am_accs)
test_var_accs = np.array(test_var_accs)
test_am_accs =np.array(test_am_accs)

fig = plt.figure()
plt.plot(np.mean(var_accs,axis=0),label="Variational Accuracy")
plt.plot(np.mean(am_accs,axis=0),label="Amortised Accuracy")
plt.plot(np.mean(test_var_accs,axis=0),label="Test Variational Accuracy")
plt.plot(np.mean(test_am_accs, axis=0),label="Test Amortised Accuracy")
plt.legend()
plt.show()

print(test_var_accs)"""
