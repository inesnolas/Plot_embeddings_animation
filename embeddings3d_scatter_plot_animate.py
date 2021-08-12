
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.core.defchararray import array
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.lines import Line2D

# https://medium.com/@pnpsegonne/animating-a-3d-scatterplot-with-matplotlib-ca4b676d4b55

# folder_path = 'C:\\Users\\madzi\\Dropbox\\QMUL\\PHD\\Plot_embeddings_animation\\TwobatchTrainingEmbeddings_plot\\'
# folder_path = 'C:\\Users\\madzi\\Dropbox\\QMUL\\PHD\\Plot_embeddings_animation\\OneBatchTrainingEmbeddings_plot\\'
# folder_path = 'C:\\Users\\madzi\\Dropbox\\QMUL\\PHD\\Plot_embeddings_animation\\315_training_set\\'
# folder_path = 'C:\\Users\\madzi\\Dropbox\\QMUL\\PHD\\Plot_embeddings_animation\\315_training_set_standardized_data_plus_batchNorm\\'
folder_path = 'C:\\Users\\madzi\\Dropbox\\QMUL\\PHD\\Plot_embeddings_animation\\315_training_set_standardized_data_only\\'
# outfilename = "315_training_set_standardized_data_plus_batchNorm"
# outfilename = "315_training_set_standardized_data_only"

# outfilename = "test"

folder_path = 'C:\\Users\\madzi\\Dropbox\\QMUL\\PHD\\Plot_embeddings_animation\\315_training_set_COSINEdistance_plus_standardizedData\\'
outfilename = "315_training_set_cosine_distance_with_standardized_data"



n_frames = 99
# import numpy as np
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.animation
# import pandas as pd
def make_color_vector_by_individual(labels):
    legend_elements = []
    unique_labels = list(set(labels))
    # color_values = np.arange(len(unique_labels))
    color_values = cm.rainbow(np.linspace(0,1,len(unique_labels)))
    dict_label_color = {}
    for i, l in enumerate(unique_labels):
        dict_label_color[l]=color_values[i]
        legend_elements.extend([Line2D([0], [0], marker='o', color=color_values[i], label=l)])
    color_vector =[dict_label_color[labels[j]] for j in range(len(labels))]
          


    return color_vector, dict_label_color, legend_elements

def group_points_per_category(embeddings, labels):

    dict_embeddings_by_category = {}
    unique_labels = list(set(labels))



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
labels = list(np.load(folder_path+"LABELS_embeddings_to_plot.csv"))
# legend = [ "PC1101_chiffchaffs_birds", "F72726_chiffchaffs_birds", "F72726_chiffchaffs_birds", "108_littleowls_birds", "0712_pipits_birds"]
# legend = [ "PC1101_chiffchaffs_birds", "F72726_chiffchaffs_birds", "F72726_chiffchaffs_birds", "0712_pipits_birds", "108_littleowls_birds" ]
# color=['b', 'k', 'r','y', 'c']
colormap = 'viridis'
colos, _, legend_elements = make_color_vector_by_individual(labels)
# legend = [ "PC1101_chiffchaffs_birds", "F72726_chiffchaffs_birds", "F72726_chiffchaffs_birds", "0712_pipits_birds", "108_littleowls_birds", "108_littleowls_birds", "108_littleowls_birds", "0212_pipits_birds", "0712_pipits_birds", "0712_pipits_birds"]
# color = ["#681F48", "#1F2368", "#68401F", "#BAECFF", "#FFF0BA", 'b', 'k', 'r','y', 'c']


for n in range(embeddings_0.shape[0]):
    scat = ax.scatter(embeddings_0[n,0], embeddings_0[n,1], embeddings_0[n,2], c=colos[n], marker=',', alpha=0.5 )# label=legend[n])
    scatters.append(scat)


ax.legend(handles=legend_elements, loc='upper right', fontsize='xx-small')
# ax.plot(embeddings_0[n,0], embeddings_0[n,2], zdir='y' )
# ax.plot(embeddings_0[n,1], embeddings_0[n,2], zdir = 'x')
# ax.plot(embeddings_0[n,0], embeddings_0[n,1], zdir = 'z')

ani =animation.FuncAnimation(fig, update_graph, n_frames, interval=15, blit=False)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
ani.save(outfilename+'.mp4', writer=writer)

# make the 2d projections!!!    https://stackoverflow.com/questions/29549905/pylab-3d-scatter-plots-with-2d-projections-of-plotted-data
# or this? https://pythonmatplotlibtips.blogspot.com/2018/01/combine-3d-two-2d-animations-in-one-figure-artistdanimation.html

plt.show()

print('stop')

    