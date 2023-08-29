import matplotlib.animation as animation
import matplotlib.pyplot as plt

class Recorder:
    def __init__(self):
        self.plots = []
        self.alphas = []
        self.color_codes = []
    def createAnimation(self):
        fig = plt.figure()
        ax = plt.axes()

        def init():
            ax.scatter([], [])

        def update(frame):
            
          
            ax.clear()
            ax.scatter(self.plots[frame][:][0],self.plots[frame][:][1], alpha=self.alphas[frame], 
                       c = self.color_codes[frame], 
                       linewidths=0)
            ax.set_xlim(0,10)
            ax.set_ylim(0,10)

        ani = animation.FuncAnimation(fig, func = update, init_func = init, frames = len(self.plots), interval = 100)
        #fig.show()
        ani.save("test.gif")