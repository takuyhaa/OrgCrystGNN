## Import packages
import os
import networkx as nx
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import numpy as np
import statistics
import math
import sklearn.metrics as metrics
import pandas as pd
plt.rcParams["font.family"] = "Dejavu sans"


## list -> print
def calc_stats(data):
    print("Size: ", len(data))
    print("Average: " , np.average(data))
    print("Max: " , np.max(data))
    print("Min: " , np.min(data))
    print("Median value: " , np.percentile(data, 50))
    print("Standard deviation: " , statistics.pstdev(data))
    

## variable -> ver name
def get_var_name(var):
    for k,v in globals().items():
        if id(v) == id(var):
            name=k
    return name


## graph data -> print the information
def check_graph(data):
    '''グラフ情報を表示'''
    print("グラフ構造:", data)
    print("グラフのキー: ", data.keys)
    print("ノード数:", data.num_nodes)
    print("エッジ数:", data.num_edges)
    print("ノードの特徴量数:", data.num_node_features)
    print("孤立したノードの有無:", data.contains_isolated_nodes())
    print("自己ループの有無:", data.contains_self_loops())
    print("====== ノードの特徴量:x ======")
    print(data['x'])
    print("====== ノードのクラス:y ======")
    print(data['y'])
    print("========= エッジ形状 =========")
    print(data['edge_index'])
    
    
## torch.data -> figure
def visgraph(data):
    nxg = to_networkx(data) # networkxのグラフに変換
    pr = nx.pagerank(nxg) # 可視化のためのページランク計算
    pr_max = np.array(list(pr.values())).max()
    draw_pos = nx.spring_layout(nxg, seed=0) # 可視化する際のノード位置
    
    index_list=[]
    for i in range(data.x.shape[0]):
#         index=(data.x[i]==1).nonzero()[0].numpy()
#         print(index)
#         index_list.append(index)
        index_list.append(data.x[i][0])

    # 図のサイズ
    plt.figure(figsize=(10, 10))

    # 描画
    nx.draw_networkx_nodes(nxg, 
                           draw_pos,
                           node_size=[v / pr_max * 1000 for v in pr.values()],
                           node_color=atomcolor(data), alpha=0.5)
    nx.draw_networkx_edges(nxg, draw_pos, arrowstyle='-', alpha=0.2)
    nx.draw_networkx_labels(nxg, draw_pos, font_size=10)
    plt.show()
    

## Graph data -> list
def atomcolor(data):
    color_list=[]
    for i in range(data.x.shape[0]):
        one_hot = data.x[i][:113]
        one_hot = one_hot.tolist()
        atom_num = one_hot.index(1)+1
        
        if atom_num==1: color_list.append("white") # H
        elif atom_num==5: color_list.append("lightpink") # B 
        elif atom_num==6: color_list.append("grey") # C
        elif atom_num==7: color_list.append("blue") # N "cornflowerblue"
        elif atom_num==8: color_list.append("red") # O
        elif atom_num==9: color_list.append("greenyellow") # F
        elif atom_num==15: color_list.append("magenta") # P
        elif atom_num==16: color_list.append("gold") # S
        elif atom_num==17: color_list.append("green") # Cl "lime"
        elif atom_num==35: color_list.append("darkgoldenrod") # Br
        elif atom_num==53: color_list.append("purple") # I
        else: color_list.append("black")
        
    return color_list


## file name -> metrics
def metric(filename):
    df=pd.read_csv(filename, index_col=0)
    target = df["target"]#df["Melting_point"]
    predict = df["prediction"]#df["Label"]
    
    size = len(target)
    r2 = metrics.r2_score(target, predict)
    rmse = np.sqrt(metrics.mean_squared_error(target, predict))
    mae = metrics.mean_absolute_error(target, predict)
    return size, r2, rmse, mae


## data -> figure
def vismetric(X, y1, y1_err,
               y2, y2_err,
               y3, y3_err,
               save=False, save_path=None, filename=None):
    
    fig = plt.figure(figsize=(18,5))
    ax1 = fig.add_subplot(1,3,1)
    ax1.errorbar(X, y1, yerr=y1_err, capsize=5, fmt='o',
                 markersize=10, ecolor='black', markeredgecolor = "black", 
                 color='black')
    ax1.set_xlabel('Training size', fontsize=15, labelpad=5)
    ax1.set_ylabel('R$^2$', fontsize=15, labelpad=6)
    ax1.xaxis.set_tick_params(labelsize=12)
    ax1.yaxis.set_tick_params(labelsize=12)
    
    ax2 = fig.add_subplot(1,3,2)
    ax2.errorbar(X, y2, yerr=y2_err, capsize=5, fmt='o',
                 markersize=10, ecolor='black', markeredgecolor = "black", 
                 color='black')
    ax2.set_xlabel('Training size', fontsize=15, labelpad=5)
    ax2.set_ylabel('RMSE', fontsize=15, labelpad=5)
    ax2.xaxis.set_tick_params(labelsize=12)
    ax2.yaxis.set_tick_params(labelsize=12)
    
    ax3 = fig.add_subplot(1,3,3)
    ax3.errorbar(X, y3, yerr=y3_err, capsize=5, fmt='o',
                 markersize=10, ecolor='black', markeredgecolor = "black", 
                 color='black')
    ax3.set_xlabel('Training size', fontsize=15, labelpad=5)
    ax3.set_ylabel('MAE', fontsize=15, labelpad=5)
    ax3.xaxis.set_tick_params(labelsize=12)
    ax3.yaxis.set_tick_params(labelsize=12)
    
    if save == True:
        fig.savefig(save_path + filename, dpi=300)


## data(target/predict) -> figure 
def viserror(file_fullpath, save=False, save_path=None):
    # Import data
    df=pd.read_csv(file_fullpath, index_col=0)
    target = df["target"]#df["Melting_point"]
    predict = df["prediction"]#df["Label"]
    file=os.path.basename(file_fullpath).rstrip(".csv")

    #Visualize
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    ax.scatter(target, predict)
    ax.plot(np.linspace(np.min(target),np.max(target),10), 
            np.linspace(np.min(target),np.max(target),10),
            c="black",
            linestyle="dashed")
    mae = metrics.mean_absolute_error(target, predict)
    
    ax.text(0.15, 0.82, "MAE="+str(round(mae,2)), 
            fontsize=15,transform=fig.transFigure)
#     ax.set_xlabel('MP$_{exp}$ (K)', fontsize=15, labelpad=10)
#     ax.set_ylabel('MP$_{pred}$ (K)', fontsize=15, labelpad=10)
    ax.set_xlabel('Band gap$_{DFT}$ (eV)', fontsize=15, labelpad=10)
    ax.set_ylabel('Band gap$_{pred}$ (eV)', fontsize=15, labelpad=10)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.set_title(file)

    if save == True:
        
        fig.savefig(save_path+file+".png", dpi=300)