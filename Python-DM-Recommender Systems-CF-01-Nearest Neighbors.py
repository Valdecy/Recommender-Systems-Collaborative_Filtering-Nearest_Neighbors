############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Data Mining
# Lesson: Recommender Systems

# Citation: 
# PEREIRA, V. (2018). Project: Recommender Systems, File: Python-DM-Recommender Systems-CF-NN.py, GitHub repository: <https://github.com/Valdecy/Recommender-Systems-CF-NN>

############################################################################

# Installing Required Libraries
import pandas as pd
import numpy  as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Function: User-User / Item-Item Similarites
def similarities(Xdata, user_in_columns = True, simil = "correlation", graph = False, mean_centering = "none"):   
    mean = "none"    
    if mean_centering == "columns":
        mean = Xdata.mean() # Missing values are discarded when calculating the mean
        Xdata = Xdata.fillna(0)
        for i in range(0, Xdata.shape[0]):
            for j in range(0, Xdata.shape[1]):
                Xdata.iloc[i, j] = Xdata.iloc[i, j] - mean[j]          
    elif mean_centering == "rows":
        mean = Xdata.T.mean() # Missing values are discarded when calculating the mean
        Xdata = Xdata.fillna(0)
        for i in range(0, Xdata.shape[0]):
            for j in range(0, Xdata.shape[1]):
                Xdata.iloc[i, j] = Xdata.iloc[i, j] - mean[i]  
    elif mean_centering == "global":
        mean = sum(Xdata.sum())/sum(Xdata.count()) # Missing values are discarded when calculating the mean
        Xdata = Xdata.fillna(0)
        for i in range(0, Xdata.shape[0]):
            for j in range(0, Xdata.shape[1]):
                Xdata.iloc[i, j] = Xdata.iloc[i, j] - mean                  
    if  user_in_columns == True:
        if simil == "correlation":
            col_col_cor = Xdata.corr()
            sim_matrix = col_col_cor
        if simil == "cosine":
            col_col_cos = cosine_similarity(np.nan_to_num(Xdata.T))
            sim_matrix = pd.DataFrame(col_col_cos, columns = Xdata.dtypes.index, index = Xdata.dtypes.index)   
    if  user_in_columns == False:
        if simil == "correlation":
            row_row_cor = Xdata.T.corr()
            sim_matrix = row_row_cor
        if simil == "cosine":
            row_row_cos = cosine_similarity(np.nan_to_num(Xdata))
            sim_matrix = pd.DataFrame(row_row_cos, columns = Xdata.T.dtypes.index, index = Xdata.T.dtypes.index)            
    if graph == True:
        f, ax = plt.subplots(figsize=(10, 10))
        cmap = sns.diverging_palette(0, 250, as_cmap =True)
        sns.heatmap(sim_matrix, cmap = cmap, square = True, linewidths = 0.5, xticklabels = sim_matrix.columns.values, yticklabels = sim_matrix.columns.values)       
    return sim_matrix, mean

# Function: K-Top Users / Items. # target = User / Item target name to show the K-Tops similarities
def k_top(Xdata, k = 5, user_in_columns = True, simil = "correlation", graph = False, mean_centering = "none", cut_off = -0.9999):   
    rank_list = [None]*1
    similarity_matrix = similarities(Xdata, user_in_columns = user_in_columns, simil = simil, graph = graph, mean_centering = mean_centering)
    sim_matrix = similarity_matrix[0]
    mean = similarity_matrix[1]    
    for j in range(0, sim_matrix.shape[0]):
        rank = sim_matrix.sort_values(sim_matrix.iloc[:,j].name, ascending = False).iloc[:,j]
        if cut_off >= -1 and cut_off <= 1:
            if (k > rank[rank >= cut_off].count() - 1):
                k = rank[rank >= cut_off].count() - 1
        rank = rank.iloc[0:k+1]
        if j == 0:
            rank_list[0] = rank
        else:
            rank_list.append(rank)
        rank = rank.iloc[0:k+1]
        rank_list[0] = rank    
    return rank_list, mean, similarity_matrix[0]

# Function: Prediction
def prediction(Xdata, rank, user_in_columns = True):
    pred = Xdata.copy(deep = True)
    rank_list = rank[0]
    mean = rank[1]
    sum_weigths = 0    
    for i in range(0, Xdata.shape[0]):
        for j in range(0, Xdata.shape[1]):
            if pd.isnull(pred.iloc[i,j]) == False:
                pred.iloc[i,j] = ""
            elif pd.isnull(pred.iloc[i,j]) == True:
                if user_in_columns == True:
                    for m in range(0, len(rank_list[j])): 
                        pred.iloc[i,j] = np.nan_to_num(pred.iloc[i,j]) + np.nan_to_num(Xdata.loc[Xdata.index[i], rank_list[j].index[m]])*rank_list[j][m]
                        if pd.isnull(Xdata.loc[Xdata.index[i], rank_list[j].index[m]]) == False:
                            sum_weigths = sum_weigths + rank_list[j][m]
                    if sum_weigths != 0:
                        pred.iloc[i,j] = pred.iloc[i,j]/sum_weigths
                    else:
                        pred.iloc[i,j] = "nan"
                    sum_weigths = 0
                elif user_in_columns == False:
                    for n in range(0, len(rank_list[i])):
                        pred.iloc[i,j] = np.nan_to_num(pred.iloc[i,j]) + np.nan_to_num(Xdata.loc[rank_list[i].index[n], Xdata.T.index[j]])*rank_list[i][n]
                        if pd.isnull(Xdata.loc[rank_list[i].index[n], Xdata.T.index[j]]) == False:
                            sum_weigths = sum_weigths + rank_list[i][n]
                    if sum_weigths != 0:
                        pred.iloc[i,j] = pred.iloc[i,j]/sum_weigths
                    else:
                        pred.iloc[i,j] = "nan"
                    sum_weigths = 0   
    if mean != "none" and user_in_columns == True and np.isscalar(mean) == False:
        for i in range(0, pred.shape[0]):
            for j in range(0, pred.shape[1]):
                pred.iloc[i, j] = pred.iloc[i, j] + mean[j]          
    elif mean != "none" and user_in_columns == False and np.isscalar(mean) == False:
        for i in range(0, pred.shape[0]):
            for j in range(0, pred.shape[1]):
                pred.iloc[i, j] = pred.iloc[i, j] + mean[i]    
    elif  mean != "none" and np.isscalar(mean) == True:
        for i in range(0, pred.shape[0]):
            for j in range(0, pred.shape[1]):
                if (isinstance(pred.iloc[i, j], str) == False):
                    pred.iloc[i, j] = pred.iloc[i, j] + mean
    return pred

######################## Part 1 - Usage ####################################

df = pd.read_csv('Python-DM-Recommender Systems-CF-01-Nearest Neighbors.txt', sep = '\t')
X = df.iloc[:, 1:]
X = X.set_index(df.iloc[:,0]) # First column as row names

# Example 1) User Similarity
rank_users = k_top(X, k = 5, user_in_columns = True, simil = "correlation", graph = True, mean_centering = "none", cut_off = 0.1)
predict_ex1 = prediction(X, rank_users, user_in_columns = True) # Prediction

# Example 2) Item Similarity
rank_items = k_top(X, k = 5, user_in_columns = False, simil = "cosine", graph = True, mean_centering = "none", cut_off = 0.5)
predict_ex2 = prediction(X, rank_items, user_in_columns = False) # Prediction


########################## End of Code #####################################
