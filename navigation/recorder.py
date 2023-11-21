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
        fig = plt.figure(figsize=(5,5), facecolor="#212121")
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
            ax.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
            ax.set_aspect('equal')
            ax.set_facecolor("#adadad")#(self.backgrounds[frame])
            string = str(round(self.frames_per_ms*frame/10)*10) + " ms"

            ax.legend(labels = [string], loc = 'lower left', bbox_to_anchor=(0,1.02,1,0.2))

        ani = animation.FuncAnimation(fig, func = update, init_func = init, frames = len(self.plots), interval = 200)
        #fig.show()
        ani.save(f"navigation/gifs/{gif_name}.gif")

class DataRec:
    def __init__(self, symbol_data, start, goal, inhibit_ranges, correct_tag, success, distance, variance):
        self.symbol_data = symbol_data
        self.start = start
        self.goal = goal
        self.inhibit_ranges = inhibit_ranges
        self.correct_tag = correct_tag
        self.success = success
        self.distance = distance
        self.variance = variance