import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

epsilon=13
samples=2000

set_mlp=np.loadtxt("../fashionmnist_relu/relu_set_mlp_sequential_fashionmnist_60000_training_samples_e20_rand0_epochs500_batchsize128.txt")
fixprob_mlp=np.loadtxt("../fashionmnist_relu_no_pruning/relu_set_mlp_sequential_fashionmnist_60000_training_samples_e20_rand0_epochs500_batchsize128.txt")
fc_mlp=np.loadtxt("../fashionmnist_alrelu_50/alternated_relu_set_mlp_sequential_fashionmnist_60000_training_samples_e20_rand0_epochs500_batchsize128.txt")
fc_mlp_2=np.loadtxt("../fashionmnist_alrelu_no_pruning/alternated_relu_set_mlp_sequential_fashionmnist_60000_training_samples_e20_rand0_epochs500_batchsize128.txt")

values_1 = []
values_2 = []
values_3 = []
values_4 = []

for i in range(1,5):
    set_mlp = set_mlp + np.loadtxt(
        "../fashionmnist_relu/relu_set_mlp_sequential_fashionmnist_60000_training_samples_e20_rand"+str(i)+"_epochs500_batchsize128.txt")
    fixprob_mlp = fixprob_mlp + np.loadtxt(
        "../fashionmnist_relu_no_pruning/relu_set_mlp_sequential_fashionmnist_60000_training_samples_e20_rand"+str(i)+"_epochs500_batchsize128.txt")
    fc_mlp = fc_mlp + np.loadtxt("../fashionmnist_alrelu_50/alternated_relu_set_mlp_sequential_fashionmnist_60000_training_samples_e20_rand"+str(i)+"_epochs500_batchsize128.txt")
    fc_mlp_2 =fc_mlp_2 +  np.loadtxt(
        "../fashionmnist_alrelu_no_pruning/alternated_relu_set_mlp_sequential_fashionmnist_60000_training_samples_e20_rand"+str(i)+"_epochs500_batchsize128.txt")
    values_1.append(np.loadtxt(
        "../fashionmnist_relu/relu_set_mlp_sequential_fashionmnist_60000_training_samples_e20_rand"+str(i)+"_epochs500_batchsize128.txt"))
    values_2.append(np.loadtxt(
        "../fashionmnist_relu_no_pruning/relu_set_mlp_sequential_fashionmnist_60000_training_samples_e20_rand"+str(i)+"_epochs500_batchsize128.txt"))
    values_3.append(np.loadtxt("../fashionmnist_alrelu_50/alternated_relu_set_mlp_sequential_fashionmnist_60000_training_samples_e20_rand"+str(i)+"_epochs500_batchsize128.txt"))
    values_4.append(np.loadtxt(
        "../fashionmnist_alrelu_no_pruning/alternated_relu_set_mlp_sequential_fashionmnist_60000_training_samples_e20_rand"+str(i)+"_epochs500_batchsize128.txt"))

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




font = { 'size'   : 6}
fig = plt.figure(figsize=(8,4))
matplotlib.rc('font', **font)
fig.subplots_adjust(wspace=0.2,hspace=0.05)

ax1=fig.add_subplot(1,2,1)
ax1.plot(set_mlp_mean[:,3]*100, label="SET-MLP ReLU + pruning", color="b", linewidth=0.5)
plt.fill_between(np.arange(0,500), set_mlp_mean[:,3] *100- set_mlp_std[:,3]*100,
                     set_mlp_mean[:,3]*100 + set_mlp_std[:,3]*100, color="b", alpha=0.1)
ax1.plot(fixprob_mlp_mean[:,3]*100, label="SET-MLP ReLU", color="m", linewidth=0.5)
plt.fill_between(np.arange(0,500), fixprob_mlp_mean[:,3]*100 - fixprob_mlp_std[:,3]*100,
                      fixprob_mlp_mean[:,3]*100 + fixprob_mlp_std[:,3]*100, color="m", alpha=0.1)
ax1.plot(fc_mlp_mean[:,3]*100, label="SET-MLP Al-ReLU + pruning", color="g", linewidth=0.5)
plt.fill_between(np.arange(0,500), fc_mlp_mean[:,3]*100 - fc_mlp_std[:,3]*100,
                     fc_mlp_mean[:,3]*100 + fc_mlp_std[:,3]*100, color="g", alpha=0.1)
ax1.plot(fc_mlp_2_mean[:,3]*100, label="SET-MLP Al-ReLU", color="orange", linewidth=0.5)
plt.fill_between(np.arange(0,500), fc_mlp_2_mean[:,3]*100 - fc_mlp_2_std[:,3]*100,
                      fc_mlp_2_mean[:,3]*100 + fc_mlp_2_std[:,3]*100, color="orange", alpha=0.1)
ax1.grid(True)
ax1.set_ylabel("FashionMNIST\nTest accuracy [%]")
ax1.set_xlabel("Epochs [#]")
ax1.legend(loc=4,fontsize=8)

ax2=fig.add_subplot(1,2,2)
ax2.plot(set_mlp_mean[:,2]*100, label="SET-MLP ReLU + pruning", color="b", linewidth=0.5)
plt.fill_between(np.arange(0,500), set_mlp_mean[:,2]*100  - set_mlp_std[:,2]*100 ,
                     set_mlp_mean[:,2]*100 + set_mlp_std[:,2]*100 , color="b", alpha=0.1)
ax2.plot(fixprob_mlp_mean[:,2]*100, label="SET-MLP ReLU", color="m", linewidth=0.5)
plt.fill_between(np.arange(0,500), fixprob_mlp_mean[:,2]*100 - fixprob_mlp_std[:,2]*100,
                 fixprob_mlp_mean[:,2]*100 + fixprob_mlp_std[:,2]*100, color="m", alpha=0.1)
ax2.plot(fc_mlp_mean[:,2]*100, label="SET-MLP Al-ReLU + pruning ", color="g", linewidth=0.5)
plt.fill_between(np.arange(0,500), fc_mlp_mean[:,2] *100- fc_mlp_std[:,2]*100,
                     fc_mlp_mean[:,2]*100 + fc_mlp_std[:,2]*100, color="g", alpha=0.1)
ax2.plot(fc_mlp_2_mean[:,2]*100, label="SET-MLP Al-ReLU", color="orange", linewidth=0.5)
plt.fill_between(np.arange(0,500), fc_mlp_2_mean[:,2]*100 - fc_mlp_2_std[:,2]*100,
                      fc_mlp_2_mean[:,2]*100 + fc_mlp_2_std[:,2]*100, color="orange", alpha=0.1)
ax2.grid(True)
ax2.set_ylabel("FashionMNIST\nTrain accuracy [%]")
ax2.set_xlabel("Epochs [#]")
ax2.legend(loc=4,fontsize=8)


plt.savefig("../Plots/SET_fashionmnist_accuracies.pdf", bbox_inches='tight')

plt.close()