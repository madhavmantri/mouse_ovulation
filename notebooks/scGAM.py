#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pygam import LinearGAM, s
import numpy as np
import pandas as pd
import time
from collections import OrderedDict
from joblib import delayed, Parallel
from sklearn.preprocessing import StandardScaler
import scanpy as sc
import seaborn as sns

# In[3]:


def gam_fit_predict(x, y, weights=None, pred_x=None, n_splines=4, spline_order=2):
    # Provide sample weights for curve fitting
    if weights is None:
        weights = np.repeat(1.0, len(x))

    # Constract weight indexing dataframe 
    use_inds = np.where(weights > 0)[0]

    # GAM fit
    gam = LinearGAM(s(0, n_splines=n_splines, spline_order=spline_order)).fit(x[use_inds], y[use_inds],
                                                           weights=weights[use_inds])

    # GAM predict
    if pred_x is None:
        pred_x = x
    y_pred = gam.predict(pred_x)

    # Calculate standard deviations
    p = gam.predict(x[use_inds])
    n = len(use_inds)
    sigma = np.sqrt(((y[use_inds] - p) ** 2).sum() / (n - 2))
    stds = (
        np.sqrt(1 + 1 / n + (pred_x - np.mean(x)) **
                2 / ((x - np.mean(x)) ** 2).sum())
        * sigma
        / 2
    )
    return y_pred, stds


# In[4]:


def compute_gene_trends(adata, var_key, genes = None, lineages_paths=None, group_by = None, lineage_branch_probs = None, use_raw = True, n_bins = 500, n_splines=4, spline_order=2, n_jobs=-1):
    '''
    adata: Anndata object of dim n_cells x n_genes
    var_key: Continous/discrete time variable key in adata.obs
    genes: Custom list of genes to calculate tread from (default: highly variable genes)
    lineage_paths: Dictionary with keys as lineage path names and values as the list of clusters (as defined by group_by variable key)
    lineage_branch_probs: n_cells x n_lineage_paths matrix with path probabilities
    group_by: Variable key to use for lineage path definitions
    use_raw: True/ False; whether to get data from raw slot
    n_bins: Number of bins for var_key binning (default: 500); set n_bins to 0 for distrete variables like time points
    n_splines: Number of splines to fit on each gene (default: 4)
    spline_order: Order of splines to fit (default: 2) 
    n_jobs: Number of jobs (Default = -1 (all))    
    '''
    
    # Compute for the entire dataset if paths not speicified
    print("Preparing...")
    
    if genes == None:
        genes = list(adata.var_names[adata.var.highly_variable])
    
    if lineages_paths is None:
        lineages_paths = {"Full" : list(adata.obs[group_by].cat.categories)}
        
    if use_raw:
        exprs_df = adata.X.to_adata().to_df()[genes]
    else: 
        exprs_df = adata.to_df()[genes]

    lineage_branch_probs = pd.DataFrame(np.ones((adata.shape[0],len(lineages_paths))), columns = lineages_paths.keys(), index = adata.obs_names)
    
    # Save results in a dictionary
    results = OrderedDict()
    
    for path in lineages_paths.keys():
        
        results[path] = OrderedDict()
        
        # Filters cells on a path and bins along the continous variable
        br_cells = adata[adata.obs[group_by].isin(lineages_paths[path])].obs_names
        br_cells = br_cells[lineage_branch_probs.loc[br_cells].loc[:, path] > 0.8]
        
        if n_bins == 0:
            bins = np.unique(adata.obs[var_key])
        else: 
            bins = np.linspace(0, adata.obs[var_key][br_cells].max(), n_bins)     
       
        # Path results onject
        results[path]["cell_ids"] = br_cells
        results[path]["trend"] = pd.DataFrame(
            0.0, index=exprs_df.columns, columns=bins
        )
        results[path]["std"] = pd.DataFrame(
            0.0, index=exprs_df.columns, columns=bins
        )

    # Compute for each path
    for path in lineages_paths.keys():
        print(path, ": ", lineages_paths[path])
        start = time.time()
        
        # Get cell IDs
        exprs_df = exprs_df.loc[results[path]["cell_ids"]]
        
        # Path and weights
        weights = lineage_branch_probs.loc[exprs_df.index, path].values
    
        bins = np.array(results[path]["trend"].columns)
        res = Parallel(n_jobs=n_jobs)(
            delayed(gam_fit_predict)(
                adata.obs[var_key][exprs_df.index].values,
                exprs_df.loc[:, gene].values,
                weights,
                bins,
                n_splines,
                spline_order
            )
            for gene in genes
        )

        # Fill in the matrices
        for i, gene in enumerate(exprs_df.columns):
            results[path]["trend"].loc[gene, :] = res[i][0]
            results[path]["std"].loc[gene, :] = res[i][1]
        end = time.time()
        print("Time for processing {}: {} minutes".format(
            path, (end - start) / 60))
        
    return results


# In[36]:


def Kmeans_cluster_gene_trends_from_lineage(trends, n_clusters = 8, scale = True, n_jobs=-1):
    """Function to cluster gene trends
    :param trends: Dataframe of gene expression trends
    :param n_jobs: Number of jobs for parallel processing
    :return: Clustering of gene trends
    """    
    if scale: 
        trends_df = pd.DataFrame(
            StandardScaler().fit_transform(trends.T).T,
            index=trends.index,
            columns=trends.columns,
        )
    else:
        trends_df = trends
        
    from sklearn import metrics
    from sklearn.cluster import KMeans
    
    # Cluster trend values
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(trends_df)
    clusters = kmeans.labels_
    clusters = pd.Series(clusters, index=trends.index)
    return clusters


# In[37]:


def DBSCAN_cluster_gene_trends_from_lineage(trends, dbscan_eps = 0.3, min_samples_per_cluster = 5, scale = True, n_jobs=-1):
    """Function to cluster gene trends
    :param trends: Dataframe of gene expression trends
    :param n_jobs: Number of jobs for parallel processing
    :return: Clustering of gene trends
    """
    if scale: 
        trends_df = pd.DataFrame(
            StandardScaler().fit_transform(trends.T).T,
            index=trends.index,
            columns=trends.columns,
        )
    else:
        trends_df = trends
    
    from sklearn.cluster import DBSCAN
    from sklearn import metrics
    # Cluster trend values
    db = DBSCAN(eps=dbscan_eps, min_samples=min_samples_per_cluster).fit(trends_df)
    clusters = db.labels_
    clusters = pd.Series(clusters, index=trends.index)
    return clusters


# In[3]:


def Agglomerative_cluster_gene_trends_from_lineage(trends, n_clusters = 8, scale = True, n_jobs=-1):
    """Function to cluster gene trends
    :param trends: Dataframe of gene expression trends
    :param n_jobs: Number of jobs for parallel processing
    :return: Clustering of gene trends
    """
    if scale: 
        trends_df = pd.DataFrame(
            StandardScaler().fit_transform(trends.T).T,
            index=trends.index,
            columns=trends.columns,
        )
    else:
        trends_df = trends
    
    from sklearn.cluster import AgglomerativeClustering as AGC
    import numpy as np
    from sklearn import metrics
    
    # Cluster trend values
    ag = AGC(n_clusters = n_clusters).fit(trends_df)
    clusters = ag.labels_
    clusters = pd.Series(clusters, index=trends.index)
    return clusters


# In[7]:


def plot_gene_expression_trends(trends, genes, scale = True):
    if scale: 
        trends_df = pd.DataFrame(
            StandardScaler().fit_transform(trends.T).T,
            index=trends.index,
            columns=trends.columns,
        ).T[genes]
    else:
        trends_df = trends.T[genes]
    trends_df = trends_df.stack().reset_index().rename(columns = {"level_0": "timepoint", "level_1":"Gene", 0: "Expression"})
    sns.lineplot(data = trends_df, x = "timepoint", y= "Expression", hue="Gene")


# In[8]:


def plot_gene_expression_trend_clusters(trends, cluster_ids, scale = True):
    if scale: 
        trends_df = pd.DataFrame(
            StandardScaler().fit_transform(trends.T).T,
            index=trends.index,
            columns=trends.columns,
        ).T
    else:
        trends_df = trends.T
    trends_df = trends_df.stack().reset_index().rename(columns = {"level_0": "timepoint", "level_1":"Gene", 0: "Expression"})
    trends_df["Cluster"] = [cluster_ids[x] for x in trends_df["Gene"]]
    sns.lineplot(data = trends_df, x = "timepoint", y= "Expression", hue="Cluster", palette="tab10")

