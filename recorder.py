import matplotlib.animation as animation
import matplotlib.pyplot as plt

class Recorder:
    def __init__(self):
        self.plots = []
    def createAnimation(self):
        fig = plt.figure()
        ani = animation.ArtistAnimation(fig, self.plots, interval = 50, blit = True)
        plt.show()