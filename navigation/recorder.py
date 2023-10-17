import matplotlib.animation as animation
import matplotlib.pyplot as plt

class Recorder:
    def __init__(self):
        self.plots = []
        self.alphas = []
        self.color_codes = []
        self.backgrounds = []
    def createAnimation(self, gif_name = "test"):
        fig = plt.figure()
        ax = plt.axes()

        def init():
            ax.scatter([], [])

        def update(frame):
            
            alphas = self.alphas[frame] if len(self.alphas[frame]) > 0 else 1
            ax.clear()
            ax.scatter(self.plots[frame][:][0],self.plots[frame][:][1], #alpha=alphas, 
                       c = self.color_codes[frame], 
                       linewidths=0)
            ax.set_xlim(0,10)
            ax.set_ylim(0,10)
            ax.set_facecolor(self.backgrounds[frame])

        ani = animation.FuncAnimation(fig, func = update, init_func = init, frames = len(self.plots), interval = 200)
        #fig.show()
        ani.save(f"navigation/gifs/{gif_name}.gif")