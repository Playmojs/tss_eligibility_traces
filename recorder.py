import matplotlib.animation as animation
import matplotlib.pyplot as plt

class Recorder:
    def __init__(self):
        self.plots = []
    def createAnimation(self):
        fig = plt.figure()
        ax = plt.axes(xlim = (0,10), ylim = (0,10))

        def update(frame):
            ax.clear()
            ax.scatter(self.plots[frame][:][0],self.plots[frame][:][1])
            ax.set_xlim(0,10)
            ax.set_ylim(0,10)

        ani = animation.FuncAnimation(fig, func = update, frames = len(self.plots), interval = 50)
        plt.show()
        ani.save("test.gif")