# okay, goal here is just to analyse the biological plausibility results correctly to get correct graphs I can use for that paper
#to start getting it into a good draft shape to be out soon. This is important to get done ASAP. Really need to be better at being FAST with this stuff...
#I have made improvements but could do faster. aim: to get all results done by 2:30... GO!
# what are the key reults I require to have made? so I'm

# okay, so these results are pretty rubbish. HOW should I deal with these...
#it's all be doing poorly... today has been SO UNFOCUSED. REALLY BAD because just procrastinating
#out of this stuff. It's just SO hard getting results including WHY collapse with the error dispersion error_dispersion_with_weight_updates
# which is also rather odd. I'm not sure what's giong on? also the no backwards weights etc are somehow wrong...
#including no proper standard experiment. Let's update that and try to figure out what's going wrong there
#which seems important to figuring it out. Annoying that error dispersion with weight updates doesn't work
#and kind of weird they all follow same pattern
#also why test variational accuracies aren't existing which is more than a bit rubbish!
#backward_weight_update			no_backward_nonlinearities
#error_dispersion_with_weight_updates	no_backward_weight_update
#error_dispersion_without_weight_updates

# okay, perfect, these results are perfect. Let's integrate them all effectively into the correct forms
#to get it sorted to create the best possible graphs for this stuff. yeah completely unsurprisingly alec is amazing! at generating
#graphs and getting them sorted to specifications. I've got to look at HOW to do it... but he gives up nice plots...
#I should try to steal his defaults.

# some better ideas include obviously the weight matrices under dispersed error connections
#and so forth so I can show that these aren't necessary... and the weight angles might be interesting to look at
#also just the direct distance between i.e. the sum of the distances to see if it does self organise or not...
#would be good to look at for this. Just do it in some case... but can run it multiple times obvs.
#would be straightforward to get those exps running too! and plot the error dispersion weights
# one key issue is we STILL don't have standard baselines, which we should have. s Let's get those asap

# alec is good at this... I'm struggling to focus to even get stuff working in the first place. IF I can generate correct results here would be GREAT
#so I can get this paper close to completion in the end would be FANTASTIC
#try to get all this stuff done with correct results tomorrow morning so you can do amortisation in the afternoon...
#going to be friday when this is done due to general multileveled failure... really quite irritating... but so it goes...
#GOT to do better. producing rubbish graphs is it's own difficulty but able to do that better
#and then keynote etc will pretty much take up a day!
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
import os
import sys
# start generic matplotlib styling implementation which would be good to have this able to be done generically.
#and would be nice got to plot these things to make them nice and clean this up.
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
