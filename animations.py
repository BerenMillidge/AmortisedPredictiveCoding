# functions to create various animations required.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate_label_beliefs(label_beliefs,anim_save_base,n,batch_num=0):
    fig, ax = plt.subplots()
    line, = ax.plot(label_beliefs[0,:,batch_num],label="Variational",alpha=0.3)
    plt.title("Variational Step " + str(0))
    plt.xlabel("Label")
    plt.ylabel("Probability")
    def init():
        line.set_ydata([np.nan] * label_beliefs.shape[1])
        return line,
    def animate(i):
        ax.cla()
        ax.plot(label_beliefs[0,:,batch_num],label="Initial",alpha=0.3)
        ax.plot(label_beliefs[i,:,batch_num],label="Variational",alpha=1,color="r")
        plt.title("Variational Step " + str(i))
        ax.legend()
        return line,
    anim = animation.FuncAnimation(fig, animate,len(label_beliefs),init_func=init, interval=1,blit=False,repeat_delay=100,save_count=len(label_beliefs))
    ax.legend()
    #plt.show()
    anim.save(anim_save_base + "epoch_"+str(n)+"label_anim.gif", fps=30)

def animate_input_predictions(input_beliefs,anim_save_base, n, batch_num=0):
    fig = plt.figure()
    plt.xticks([])
    plt.yticks([])
    im = plt.imshow(input_beliefs[0,:,batch_num].reshape(28,28), animated=True, aspect='auto')
    plt.title("Variational Epoch 0")
    def animate(i):
    	im.set_array(input_beliefs[i,:,batch_num].reshape(28,28))
    	title = plt.title('Variational Epoch ' +str(i))
    	return im,

    length = len(input_beliefs)
    plt.subplots_adjust(wspace=0, hspace=0)
    anim = animation.FuncAnimation(fig, animate,len(label_beliefs), interval=1,blit=False,repeat_delay=100,save_count=len(label_beliefs))
    #plt.show()
    #anim.save(anim_save_base + "epoch_"+str(n)+"image_anim.mp4",writer="ffmpeg", fps=30, extra_args=['-vcodec', 'libx264'])
    anim.save(anim_save_base + "epoch_"+str(n)+"image_anim.gif",fps=30)
