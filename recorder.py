import matplotlib.animation as animation
import matplotlib.pyplot as plt

class Recorder:
    def __init__(self):
        self.plots = []
    def createAnimation(self):
        fig = plt.figure()
        ax = plt.axes(xlim = (0,10), ylim = (0,10))
        scat = ax.scatter(self.plots[0][:][0], self.plots[0][:][1])

        def update(frame):
            scat.set_offsets(self.plots[frame])

        ani = animation.FuncAnimation(fig, func = update, frames = len(self.plots), interval = 50)
        plt.show()