import matplotlib.animation as animation
import matplotlib.pyplot as plt

class Recorder:
    def __init__(self, frames_per_ms = 1):
        self.plots = []
        self.alphas = []
        self.color_codes = []
        self.backgrounds = []
        self.frames_per_ms = frames_per_ms
    def createAnimation(self, gif_name = "test"):
        fig = plt.figure()
        ax = plt.axes()

        def init():
            ax.scatter([], [])

        def update(frame):
            
            alphas = self.alphas[frame] if len(self.alphas[frame]) > 0 else 1
            ax.clear()
            ax.scatter(self.plots[frame][:][0],self.plots[frame][:][1], alpha=alphas, 
                       c = self.color_codes[frame], 
                       linewidths=0)
            ax.set_xlim(0,10)
            ax.set_ylim(0,10)
            ax.set_facecolor(self.backgrounds[frame])
            string = str(round(self.frames_per_ms*frame/10)*10) + " ms"

            ax.legend(labels = [string], loc = 'upper right')

        ani = animation.FuncAnimation(fig, func = update, init_func = init, frames = len(self.plots), interval = 1)
        #fig.show()
        ani.save(f"gifs/{gif_name}.gif")