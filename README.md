# Collaborative Filtering - Nearest Neighbors
Collaborative Filtering: This approach builds a model from past behaviors, comparing items or users trough ratings, and in this case the Nearest Neighbors (memory based) is used to calculate the k-top similar users/items (cosine or pearson similarity). The function returns, the k-top similar users/items, the mean and the similarity matrix.

* Xdata = Dataset Attributes. A matrix with users ratings about a set of items.

* k = Up to k-top similarities (cosine similarity or pearson correlation) that are greater or equal the cut_off value. The default value is 5.

* user_in_columns = Boolean that indicates if the user is in the column (user_in_column = True) then a user-user similarity is made, if (user_in_column = False) then an item-item similarity is performed instead. The default value is True.

* simil = "cosine", "correlation". If "cosine" is chosen then a cosine similarity is performed, and if "correlation" is chosen then a pearson correlation is performed. The default value is "correlation".

* graph = Boolean that indicates if the similarity matrix will be displayed (graph = True) or not (graph = False). The default value is True.

* mean_centering = "none", "row", "column", "global". If "none" is selected then no centering is made, if "row" is selected then a row mean centering is performed,  if "column" is selected then a column mean centering is performed and if "global" is selected then a global centering (matrix mean) is performed. The default value is "none".

* cut_off = Value between -1 and 1 that filter similar item according to a certain threshold value. The default value is -0.9999.

* Finnaly a prediction function - prediction( ) - is also included.

# Recommender System Library
Try [pyRecommenderSystem](https://github.com/Valdecy/pyRecommenderSystem): A Recommender System Library
