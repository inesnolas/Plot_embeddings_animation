
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.core.defchararray import array
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# https://medium.com/@pnpsegonne/animating-a-3d-scatterplot-with-matplotlib-ca4b676d4b55


folder_path = 'C:\\Users\\madzi\\Dropbox\\QMUL\\PHD\\JTs_code_plot_embeddings\\embeddings\\'

# import numpy as np
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.animation
# import pandas as pd


def update_graph(num):
    embeddings_at_n = np.load(folder_path+"batch_embeddings_at_"+str(num)+'.csv')

    for n in range(embeddings_at_n.shape[0]):
        scatters[n]._offsets3d = (embeddings_at_n[n,0:1], embeddings_at_n[n,1:2], embeddings_at_n[n,2:])
    title.set_text('3D Test, time={}'.format(num))
    

# def init_graph():
#     for scat in graph:
#         scat.set_offsets([])
#     return graph

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('3D Test')


scatters = []
embeddings_0 = np.load(folder_path+"batch_embeddings_at_"+str(0)+'.csv')
legend = [ "PC1101_chiffchaffs_birds", "F72726_chiffchaffs_birds", "F72726_chiffchaffs_birds", "108_littleowls_birds", "0712_pipits_birds"]
colo = c=['b', 'k', 'r','y', 'c']

for n in range(embeddings_0.shape[0]):
    scat = ax.scatter(embeddings_0[n,0], embeddings_0[n,1], embeddings_0[n,2], c=colo[n], label=legend[n])
    scatters.append(scat)

ax.legend()


ani =animation.FuncAnimation(fig, update_graph, 49, interval=10, blit=False)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
ani.save('3d-scatted-animated.mp4', writer=writer)



plt.show()


    