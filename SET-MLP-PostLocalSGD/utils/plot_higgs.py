import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
epsilon=13
samples=2000

set_mlp=np.loadtxt("../higgs_relu_std/relu_set_mlp_sequential_higgs_60000_training_samples_e10_rand0_epochs500_batchsize128.txt")
fixprob_mlp=np.loadtxt("../higgs_relu_std_no_pruning/relu_set_mlp_sequential_higgs_60000_training_samples_e10_rand0_epochs500_batchsize128.txt")
fc_mlp=np.loadtxt("../higgs_alrelu_5/alrelu_set_mlp_sequential_higgs_60000_training_samples_e10_rand0_epochs500_batchsize128.txt")
fc_mlp_2=np.loadtxt("../higgs_alrelu_5_no_pruning/alrelu_set_mlp_sequential_higgs_60000_training_samples_e10_rand0_epochs500_batchsize128.txt")

values_1 = []
values_2 = []
values_3 = []
values_4 = []

for i in range(1,5):
    set_mlp = set_mlp + np.loadtxt("../higgs_relu_std/relu_set_mlp_sequential_higgs_60000_training_samples_e10_rand"+str(i)+"_epochs500_batchsize128.txt")
    fixprob_mlp = fixprob_mlp + np.loadtxt("../higgs_relu_std_no_pruning/relu_set_mlp_sequential_higgs_60000_training_samples_e10_rand"+str(i)+"_epochs500_batchsize128.txt")
    fc_mlp = fc_mlp + np.loadtxt("../higgs_alrelu_5/alrelu_set_mlp_sequential_higgs_60000_training_samples_e10_rand"+str(i)+"_epochs500_batchsize128.txt")
    fc_mlp_2 =fc_mlp_2 +  np.loadtxt("../higgs_alrelu_5_no_pruning/alrelu_set_mlp_sequential_higgs_60000_training_samples_e10_rand"+str(i)+"_epochs500_batchsize128.txt")
    values_1.append(np.loadtxt("../higgs_relu_std/relu_set_mlp_sequential_higgs_60000_training_samples_e10_rand"+str(i)+"_epochs500_batchsize128.txt"))
    values_2.append(np.loadtxt("../higgs_relu_std_no_pruning/relu_set_mlp_sequential_higgs_60000_training_samples_e10_rand"+str(i)+"_epochs500_batchsize128.txt"))
    values_3.append(np.loadtxt("../higgs_alrelu_5/alrelu_set_mlp_sequential_higgs_60000_training_samples_e10_rand"+str(i)+"_epochs500_batchsize128.txt"))
    values_4.append(np.loadtxt("../higgs_alrelu_5_no_pruning/alrelu_set_mlp_sequential_higgs_60000_training_samples_e10_rand"+str(i)+"_epochs500_batchsize128.txt"))

values_1 = np.asarray(values_1)
values_2 = np.asarray(values_2)
values_3 = np.asarray(values_3)
values_4 = np.asarray(values_4)

set_mlp_mean= set_mlp /5
fixprob_mlp_mean= fixprob_mlp/5
fc_mlp_mean= fc_mlp /5
fc_mlp_2_mean = fc_mlp_2/5

set_mlp_std = values_1.std(axis=0)
fixprob_mlp_std =values_2.std(axis=0)
fc_mlp_std =values_3.std(axis=0)
fc_mlp_2_std= values_4.std(axis=0)




font = { 'size'   : 10}
fig = plt.figure(figsize=(15,4))
matplotlib.rc('font', **font)
fig.subplots_adjust(wspace=0.2,hspace=0.05)

ax1=fig.add_subplot(1,3,1)
ax1.plot(set_mlp_mean[:,3]*100, label="SET-MLP ReLU + pruning", color="b", linewidth=0.3)
plt.fill_between(np.arange(0,500), set_mlp_mean[:,3] *100- set_mlp_std[:,3]*100,
                     set_mlp_mean[:,3]*100 + set_mlp_std[:,3]*100, color="b", alpha=0.15)
ax1.plot(fixprob_mlp_mean[:,3]*100, label="SET-MLP ReLU", color="m", linewidth=0.3)
plt.fill_between(np.arange(0,500), fixprob_mlp_mean[:,3]*100 - fixprob_mlp_std[:,3]*100,
                      fixprob_mlp_mean[:,3]*100 + fixprob_mlp_std[:,3]*100, color="m", alpha=0.15)
ax1.plot(fc_mlp_mean[:,3]*100, label="SET-MLP Al-ReLU + pruning", color="g", linewidth=0.3)
plt.fill_between(np.arange(0,500), fc_mlp_mean[:,3]*100 - fc_mlp_std[:,3]*100,
                     fc_mlp_mean[:,3]*100 + fc_mlp_std[:,3]*100, color="g", alpha=0.15)
ax1.plot(fc_mlp_2_mean[:,3]*100, label="SET-MLP Al-ReLU", color="orange", linewidth=0.3)
plt.fill_between(np.arange(0,500), fc_mlp_2_mean[:,3]*100 - fc_mlp_2_std[:,3]*100,
                      fc_mlp_2_mean[:,3]*100 + fc_mlp_2_std[:,3]*100, color="orange", alpha=0.15)
ax1.grid(color='lightgray', ls = ':', lw = 0.25)
ax1.set_ylabel("HIGGS\nTest accuracy [%]")
ax1.set_xlabel("Epochs [#]")
ax1.legend(loc=4,fontsize=10)

axins = zoomed_inset_axes(ax1, 2.5, loc=7)
axins.plot(set_mlp_mean[:,3]*100, color='b', linewidth=0.3)
axins.fill_between(np.arange(0,500), fixprob_mlp_mean[:,3]*100 - fixprob_mlp_std[:,3]*100,
                      fixprob_mlp_mean[:,3]*100 + fixprob_mlp_std[:,3]*100, color="m", alpha=0.15)
axins.plot(fixprob_mlp_mean[:,3]*100, color='m', linewidth=0.3)
axins.fill_between(np.arange(0,500), fixprob_mlp_mean[:,3]*100 - fixprob_mlp_std[:,3]*100,
                      fixprob_mlp_mean[:,3]*100 + fixprob_mlp_std[:,3]*100, color="m", alpha=0.15)
axins.plot(fc_mlp_mean[:,3]*100, color='g', linewidth=0.3)
axins.fill_between(np.arange(0,500), fc_mlp_mean[:,3]*100 - fc_mlp_std[:,3]*100,
                     fc_mlp_mean[:,3]*100 + fc_mlp_std[:,3]*100, color="g", alpha=0.15)
axins.plot(fc_mlp_2_mean[:,3]*100, color='orange', linewidth=0.3)
axins.fill_between(np.arange(0,500), fc_mlp_2_mean[:,3]*100 - fc_mlp_2_std[:,3]*100,
                      fc_mlp_2_mean[:,3]*100 + fc_mlp_2_std[:,3]*100, color="orange", alpha=0.15)
x1, x2, y1, y2 = 375, 500, 72.5, 74 # specify the limits
axins.set_xlim(x1, x2) # apply the x-limits
axins.set_ylim(y1, y2) # apply the y-limits
plt.yticks(visible=False)
plt.xticks(visible=False)
mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="0.5")

ax2=fig.add_subplot(1,3,2)
ax2.plot(set_mlp_mean[:,2]*100, label="SET-MLP ReLU + pruning", color="b", linewidth=0.3)
plt.fill_between(np.arange(0,500), set_mlp_mean[:,2]*100  - set_mlp_std[:,2]*100 ,
                     set_mlp_mean[:,2]*100 + set_mlp_std[:,2]*100 , color="b", alpha=0.15)
ax2.plot(fixprob_mlp_mean[:,2]*100, label="SET-MLP ReLU", color="m", linewidth=0.3)
plt.fill_between(np.arange(0,500), fixprob_mlp_mean[:,2]*100 - fixprob_mlp_std[:,2]*100,
                 fixprob_mlp_mean[:,2]*100 + fixprob_mlp_std[:,2]*100, color="m", alpha=0.15)
ax2.plot(fc_mlp_mean[:,2]*100, label="SET-MLP Al-ReLU + pruning ", color="g", linewidth=0.3)
plt.fill_between(np.arange(0,500), fc_mlp_mean[:,2] *100- fc_mlp_std[:,2]*100,
                     fc_mlp_mean[:,2]*100 + fc_mlp_std[:,2]*100, color="g", alpha=0.15)
ax2.plot(fc_mlp_2_mean[:,2]*100, label="SET-MLP Al-ReLU", color="orange", linewidth=0.3)
plt.fill_between(np.arange(0,500), fc_mlp_2_mean[:,2]*100 - fc_mlp_2_std[:,2]*100,
                      fc_mlp_2_mean[:,2]*100 + fc_mlp_2_std[:,2]*100, color="orange", alpha=0.15)
ax2.grid(color='lightgray', ls = ':', lw = 0.25)
ax2.set_ylabel("HIGGS\nTrain accuracy [%]")
ax2.set_xlabel("Epochs [#]")
ax2.legend(loc=4,fontsize=10)

ax3=fig.add_subplot(1,3,3)
ax3.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
langs = ['Dense MLP', 'SET-MLP', 'SET-MLP \n+\n importance pruning']
students = [2033002, 50165, 9992]
colors = ['b', 'orange', 'r']

ax3.grid(color='lightgray', ls = ':', lw = 0.25, zorder=0)
ax3.bar(langs,students, color=colors, zorder=3)


plt.savefig("../Plots/SET_higgs_plots.pdf", bbox_inches='tight')

plt.close()