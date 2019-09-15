from __future__ import division
import os
import os.path
from scipy import linalg
import scipy.sparse as sparse
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

def gen_tr_ho_networks(A_orig, alpha, alpha_):

    """ 
    This function constructs the holdout and training adjacency matrix uniformly sampled from the original adjacency matrix.

    Input and Parameters:
    -------
    A_orig: the original adjacency matrix

    Returns:
    -------
    A_ho : the holdout adjacency matrix
    A_tr : training adjacency matrix
    alpha : the sample probability used for sampling from the original matrix to create the holdout matrix
    alpha_ : the sample probability used for sampling from the holdout matrix to create the training matrix

    Examples:
    -------
    >>> A_ho, A_tr = gen_tr_ho_networks(A_orig, alpha, alpha_)
    """

    A_ho = 1*(np.triu(A_orig,1)==1)
    rows_one, cols_one = np.where(np.triu(A_ho,1))
    ones_prob_samp = np.random.binomial(1, size=len(rows_one), p=alpha)
    A_ho[rows_one, cols_one] = ones_prob_samp
    A_ho = A_ho + A_ho.T

    A_tr = 1*(np.triu(A_ho,1)==1)
    rows_one, cols_one = np.where(np.triu(A_tr,1))
    ones_prob_samp = np.random.binomial(1, size=len(rows_one), p=alpha_)
    A_tr[rows_one, cols_one] = ones_prob_samp
    A_tr = A_tr + A_tr.T
    return A_ho, A_tr

def adj_to_nodes_edges(A):
    
    """ 
    This function change adjacency matrix to list of nodes and edges.

    Input and Parameters:
    -------
    A: the adjacency matrix

    Returns:
    -------
    nodes: node list of the given network
    edges: edge list of the given network

    Examples:
    -------
    >>> nodes, edges = adj_to_nodes_edges(A)
    """
    
    num_nodes = A.shape[0]
    nodes = range(num_nodes)
    edges = np.where(np.triu(A,1))
    row = edges[0]
    col = edges[1]
    edges = np.vstack((row,col)).T
    return nodes, edges


def sample_true_false_edges(A_orig, A_tr, A_ho):  
    
    """ 
    This function creates the training and holdout samples.

    Input and Parameters:
    -------
    A: the adjacency matrix

    Returns:
    -------
    nodes: node list of the given network
    edges: edge list of the given network

    Examples:
    -------
    >>> nodes, edges = adj_to_nodes_edges(A)
    """
    
    nodes, edge_tr = adj_to_nodes_edges(A_tr)
    nsim_id = 0
    np.random.seed(nsim_id)

    A_diff = A_ho - A_tr
    e_diff = sparse.find(sparse.triu(A_diff,1)) # true candidates
    A_ho_aux = -1*A_ho + 1
    ne_ho = sparse.find(sparse.triu(A_ho_aux,1)) # false candidates
    Nsamples = 10000 # number of samples
    edge_t = [] # list of true edges (positive samples)
    edge_f = [] # list of false edges (negative samples)
    for ll in range(Nsamples):
	    edge_t_idx_aux = np.random.randint(len(e_diff[0]))
	    edge_f_idx_aux = np.random.randint(len(ne_ho[0]))
	    edge_t.append((e_diff[0][edge_t_idx_aux],e_diff[1][edge_t_idx_aux]))
	    edge_f.append((ne_ho[0][edge_f_idx_aux],ne_ho[1][edge_f_idx_aux]))
	
	# store for later use
    if not os.path.isdir("./edge_tf_tr/"):
	    os.mkdir("./edge_tf_tr/")
    np.savetxt("./edge_tf_tr/edge_t.txt",edge_t,fmt='%u')
    np.savetxt("./edge_tf_tr/edge_f.txt",edge_f,fmt='%u')
    
    A_diff = A_orig - A_ho
    e_diff = sparse.find(sparse.triu(A_diff,1)) # true candidates
    A_orig_aux = -1*A_orig + 1
    ne_orig = sparse.find(sparse.triu(A_orig_aux,1)) # false candidates
    Nsamples = 10000 # number of samples
    edge_t = [] # list of true edges (positive samples)
    edge_f = [] # list of false edges (negative samples)
    for ll in range(Nsamples):
        edge_t_idx_aux = np.random.randint(len(e_diff[0]))
        edge_f_idx_aux = np.random.randint(len(ne_orig[0]))
        edge_t.append((e_diff[0][edge_t_idx_aux],e_diff[1][edge_t_idx_aux]))
        edge_f.append((ne_orig[0][edge_f_idx_aux],ne_orig[1][edge_f_idx_aux]))
    
    # store for later use
    if not os.path.isdir("./edge_tf_ho/"):
        os.mkdir("./edge_tf_ho/")
    np.savetxt("./edge_tf_ho/edge_t.txt",edge_t,fmt='%u')
    np.savetxt("./edge_tf_ho/edge_f.txt",edge_f,fmt='%u')


def gen_topol_feats(A_orig, A, edge_s): 
    
    """ 
    This function generates the topological features for matrix A (A_tr or A_ho) over edge samples edge_s (edge_tr or edge_ho).

    Input and Parameters:
    -------
    A: the training or holdout adjacency matrix that the topological features are going to be computed over
    A_orig: the original adjacency matrix
    edge_s: the sample set of training or holdout edges that the topological features are going to be computed over

    Returns:
    -------
    df_feat: data frame of features

    Examples:
    -------
    >>> gen_topol_feats(A_orig, A_tr, edge_tr)
    >>> gen_topol_feats(A_orig, A_ho, edge_ho)
    """

    _, edges = adj_to_nodes_edges(A)    
    nodes = [int(iii) for iii in range(A.shape[0])]
    N = len(nodes)
    if len(edges.shape)==1:
        edges = [(int(iii),int(jjj)) for iii,jjj in [edges]]
    else:
        edges = [(int(iii),int(jjj)) for iii,jjj in edges]

    # define graph
    G=nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # average degree (AD)
    ave_deg_net = np.sum(A)/A.shape[0]
    # variance of degree distribution (VD)
    var_deg_net = np.sqrt(np.sum(np.square(np.sum(A,axis = 0)-ave_deg_net))/(A.shape[0]-1))
    # average (local) clustering coefficient (ACC)
    ave_clust_net = nx.average_clustering(G)
    
    # samples chosen - features
    edge_pairs_f_i = edge_s[:,0]
    edge_pairs_f_j = edge_s[:,1]
    

    # local number of triangles for i and j (LNT_i, LNT_j)  
    numtriang_nodes_obj = nx.triangles(G)
    numtriang_nodes = []
    for nn in range(len(nodes)):
        numtriang_nodes.append(numtriang_nodes_obj[nn])
     
    numtriang1_edges = []
    numtriang2_edges = []
    for ee in range(len(edge_s)):
        numtriang1_edges.append(numtriang_nodes[edge_s[ee][0]])
        numtriang2_edges.append(numtriang_nodes[edge_s[ee][1]])
         
    # Page rank values for i and j (PR_i, PR_j)
    page_rank_nodes_obj = nx.pagerank(G)
    page_rank_nodes = []
    for nn in range(len(nodes)):
        page_rank_nodes.append(page_rank_nodes_obj[nn])
        
    page_rank1_edges = []
    page_rank2_edges = []
    for ee in range(len(edge_s)):
        page_rank1_edges.append(page_rank_nodes[edge_s[ee][0]])
        page_rank2_edges.append(page_rank_nodes[edge_s[ee][1]])
        
    # j-th entry of the personalized page rank of node i (PPR)
    page_rank_pers_nodes = []
    hot_vec = {}
    for nn in range(len(nodes)):
        hot_vec[nn] = 0
    for nn in range(len(nodes)):
        hot_vec_copy = hot_vec.copy()
        hot_vec_copy[nn] = 1 
        page_rank_pers_nodes.append(nx.pagerank(G,personalization=hot_vec_copy))

    page_rank_pers_edges = []
    for ee in range(len(edge_s)):
        page_rank_pers_edges.append(page_rank_pers_nodes[edge_s[ee][0]][edge_s[ee][1]])

    # local clustering coefficients for i and j (LCC_i, LCC_j)
    clust_nodes_obj = nx.clustering(G)
    clust_nodes = []
    for nn in range(len(nodes)):
        clust_nodes.append(clust_nodes_obj[nn])
    
    clust1_edges = []
    clust2_edges = []
    for ee in range(len(edge_s)):
        clust1_edges.append(clust_nodes[edge_s[ee][0]])
        clust2_edges.append(clust_nodes[edge_s[ee][1]])
        
    # average neighbor degrees for i and j (AND_i, AND_j)
    ave_neigh_deg_nodes_obj = nx.average_neighbor_degree(G)
    ave_neigh_deg_nodes = []
    for nn in range(len(nodes)):
        ave_neigh_deg_nodes.append(ave_neigh_deg_nodes_obj[nn])
    
    ave_neigh_deg1_edges = []
    ave_neigh_deg2_edges = []
    for ee in range(len(edge_s)):
        ave_neigh_deg1_edges.append(ave_neigh_deg_nodes[edge_s[ee][0]])
        ave_neigh_deg2_edges.append(ave_neigh_deg_nodes[edge_s[ee][1]])
   
    # degree centralities for i and j (DC_i, DC_j)
    deg_cent_nodes_obj = nx.degree_centrality(G)
    deg_cent_nodes = []
    for nn in range(len(nodes)):
        deg_cent_nodes.append(deg_cent_nodes_obj[nn])
     
    deg_cent1_edges = []
    deg_cent2_edges = []
    for ee in range(len(edge_s)):
        deg_cent1_edges.append(deg_cent_nodes[edge_s[ee][0]])
        deg_cent2_edges.append(deg_cent_nodes[edge_s[ee][1]])
  
	# eigenvector centralities for i and j (EC_i, EC_j)
    tr = 1
    toler = 1e-6
    while tr == 1:
        try:
            eig_cent_nodes_obj = nx.eigenvector_centrality(G,tol = toler)
            tr = 0
        except:
            toler = toler*1e1
    
    eig_cent_nodes = []
    for nn in range(len(nodes)):
        eig_cent_nodes.append(eig_cent_nodes_obj[nn])
     
    eig_cent1_edges = []
    eig_cent2_edges = []
    for ee in range(len(edge_s)):
        eig_cent1_edges.append(eig_cent_nodes[edge_s[ee][0]])
        eig_cent2_edges.append(eig_cent_nodes[edge_s[ee][1]])

    # Katz centralities for i and j (KC_i, KC_j)
    ktz_cent_nodes_obj = nx.katz_centrality_numpy(G)
    ktz_cent_nodes = []
    for nn in range(len(nodes)):
        ktz_cent_nodes.append(ktz_cent_nodes_obj[nn])
    
    ktz_cent1_edges = []
    ktz_cent2_edges = []
    for ee in range(len(edge_s)):
        ktz_cent1_edges.append(ktz_cent_nodes[edge_s[ee][0]])
        ktz_cent2_edges.append(ktz_cent_nodes[edge_s[ee][1]])
         
    # Jaccard’s coefficient of neighbor sets of i, j (JC)
    jacc_coeff_obj = nx.jaccard_coefficient(G,edge_s)
    jacc_coeff_edges = []
    for uu,vv,jj in jacc_coeff_obj:
        jacc_coeff_edges.append([uu,vv,jj])   
    df_jacc_coeff = pd.DataFrame(jacc_coeff_edges, columns=['i','j','jacc_coeff'])
    df_jacc_coeff['ind'] = df_jacc_coeff.index

    # resource allocation index of i, j (RA)
    res_alloc_ind_obj = nx.resource_allocation_index(G, edge_s)
    res_alloc_ind_edges = []
    for uu,vv,jj in res_alloc_ind_obj:
        res_alloc_ind_edges.append([uu,vv,jj])
    df_res_alloc_ind = pd.DataFrame(res_alloc_ind_edges, columns=['i','j','res_alloc_ind'])    
    df_res_alloc_ind['ind'] = df_res_alloc_ind.index
  
  	# Adamic/Adar index of i, j (AA)
    adam_adar_obj =  nx.adamic_adar_index(G, edge_s)
    adam_adar_edges = []
    for uu,vv,jj in adam_adar_obj:
        adam_adar_edges.append([uu,vv,jj])
    df_adam_adar = pd.DataFrame(adam_adar_edges, columns=['i','j','adam_adar'])
    df_adam_adar['ind'] = df_adam_adar.index
    
    df_merge = pd.merge(df_jacc_coeff,df_res_alloc_ind, on=['ind','i','j'], sort=False)
    df_merge = pd.merge(df_merge,df_adam_adar, on=['ind','i','j'], sort=False)

    # preferential attachment (degree product) of i, j (PA)
    pref_attach_obj = nx.preferential_attachment(G, edge_s)
    pref_attach_edges = []
    for uu,vv,jj in pref_attach_obj:
        pref_attach_edges.append([uu,vv,jj])
    df_pref_attach = pd.DataFrame(pref_attach_edges, columns=['i','j','pref_attach'])
    df_pref_attach['ind'] = df_pref_attach.index
                
    # global features:
    # similarity of connections in the graph with respect to the node degree
    # degree assortativity (DA)
    deg_ass_net = nx.degree_assortativity_coefficient(G)
    # transitivity: fraction of all possible triangles present in G
    # network transitivity (clustering coefficient) (NT)
    transit_net = nx.transitivity(G)  
    # network diameter (ND)
    try:
        diam_net = nx.diameter(G)
    except:
        diam_net = np.inf
        
    ave_deg_net = [ave_deg_net for ii in range(10000)]
    var_deg_net = [var_deg_net for ii in range(10000)]
    ave_clust_net = [ave_clust_net for ii in range(10000)]
    deg_ass_net = [deg_ass_net for ii in range(10000)]
    transit_net = [transit_net for ii in range(10000)]
    diam_net = [diam_net for ii in range(10000)]
    com_ne = []
    for ee in range(len(edge_s)):
        com_ne.append(len(sorted(nx.common_neighbors(G,edge_s[ee][0],edge_s[ee][1]))))
          
    # closeness centralities for i and j (CC_i, CC_j)
    closn_cent_nodes_obj = nx.closeness_centrality(G)
    closn_cent_nodes = []
    for nn in range(len(nodes)):
        closn_cent_nodes.append(closn_cent_nodes_obj[nn])
      
    closn_cent1_edges = []
    closn_cent2_edges = []
    for ee in range(len(edge_s)):
        closn_cent1_edges.append(closn_cent_nodes[edge_s[ee][0]])
        closn_cent2_edges.append(closn_cent_nodes[edge_s[ee][1]])
          
    # shortest path between i, j (SP)        
    short_Mat_aux = nx.shortest_path_length(G)
    short_Mat={}
    for ss in range(N):
        value = next(short_Mat_aux)
        short_Mat[value[0]] = value[1]   
    short_path_edges = []
    for ee in range(len(edge_s)):
        if edge_s[ee][1] in short_Mat[edge_s[ee][0]].keys():
            short_path_edges.append(short_Mat[edge_s[ee][0]][edge_s[ee][1]])  
        else:
            short_path_edges.append(np.inf)
            
    # load centralities for i and j (LC_i, LC_j)
    load_cent_nodes_obj = nx.load_centrality(G,normalized=True)
    load_cent_nodes = []
    for nn in range(len(nodes)):
        load_cent_nodes.append(load_cent_nodes_obj[nn])
    
    load_cent1_edges = []
    load_cent2_edges = []
    for ee in range(len(edge_s)):
        load_cent1_edges.append(load_cent_nodes[edge_s[ee][0]])
        load_cent2_edges.append(load_cent_nodes[edge_s[ee][1]])

    # shortest-path betweenness centralities for i and j (SPBC_i, SPBC_j)
    betw_cent_nodes_obj = nx.betweenness_centrality(G,normalized=True)
    betw_cent_nodes = []
    for nn in range(len(nodes)):
        betw_cent_nodes.append(betw_cent_nodes_obj[nn])
    
    betw_cent1_edges = []
    betw_cent2_edges = []
    for ee in range(len(edge_s)):
        betw_cent1_edges.append(betw_cent_nodes[edge_s[ee][0]])
        betw_cent2_edges.append(betw_cent_nodes[edge_s[ee][1]])
           
    neigh_ = {}
    for nn in range(len(nodes)):
        neigh_[nn] = np.where(A[nn,:])[0]
    
    df_pref_attach = []
    for ee in range(len(edge_s)):
        df_pref_attach.append(len(neigh_[edge_s[ee][0]])*len(neigh_[edge_s[ee][1]]))
    
    U, sig, V = np.linalg.svd(A, full_matrices=False)
    S = np.diag(sig)
    Atilda = np.dot(U, np.dot(S, V))
    Atilda = np.array(Atilda)
    
    f_mean = lambda x: np.mean(x) if len(x)>0 else 0
    # entry i, j in low rank approximation (LRA) via singular value decomposition (SVD)
    svd_edges = []
    # dot product of columns i and j in LRA via SVD for each pair of nodes i, j
    svd_edges_dot = []
    # average of entries i and j’s neighbors in low rank approximation
    svd_edges_mean = []
    for ee in range(len(edge_s)):
        svd_edges.append(Atilda[edge_s[ee][0],edge_s[ee][1]])
        svd_edges_dot.append(np.inner(Atilda[edge_s[ee][0],:],Atilda[:,edge_s[ee][1]]))
        svd_edges_mean.append(f_mean(Atilda[edge_s[ee][0],neigh_[edge_s[ee][1]]]))        
    
    # Leicht-Holme-Newman index of neighbor sets of i, j (LHN)
    f_LHN = lambda num,den: 0 if (num==0 and den==0) else float(num)/den 
    LHN_edges = [f_LHN(num,den) for num,den in zip(np.array(com_ne),np.array(df_pref_attach))]
    
    U, sig, V = np.linalg.svd(A)
    S = linalg.diagsvd(sig, A.shape[0], A.shape[1])
    S_trunc = S.copy()
    S_trunc[S_trunc < sig[int(np.ceil(np.sqrt(A.shape[0])))]] = 0
    Atilda = np.dot(np.dot(U, S_trunc), V)
    Atilda = np.array(Atilda)
    
    f_mean = lambda x: np.mean(x) if len(x)>0 else 0
    # an approximation of LRA (LRA-approx)
    svd_edges_approx = []
    # an approximation of dLRA (dLRA-approx)
    svd_edges_dot_approx = []
    # an approximation of mLRA (mLRA-approx)
    svd_edges_mean_approx = []
    for ee in range(len(edge_s)):
        svd_edges_approx.append(Atilda[edge_s[ee][0],edge_s[ee][1]])
        svd_edges_dot_approx.append(np.inner(Atilda[edge_s[ee][0],:],Atilda[:,edge_s[ee][1]]))
        svd_edges_mean_approx.append(f_mean(Atilda[edge_s[ee][0],neigh_[edge_s[ee][1]]])) 

    # number of nodes (N)
    num_nodes = A_orig.shape[0]
    # number of observed edges (OE)
    num_edges = int(np.sum(A)/2)
    
    # construct a dictionary of the features
    d = {'i':edge_pairs_f_i, 'j':edge_pairs_f_j, 'com_ne':com_ne, 'ave_deg_net':ave_deg_net, \
         'var_deg_net':var_deg_net, 'ave_clust_net':ave_clust_net, 'num_triangles_1':numtriang1_edges, 'num_triangles_2':numtriang2_edges, \
         'page_rank_pers_edges':page_rank_pers_edges, 'pag_rank1':page_rank1_edges, 'pag_rank2':page_rank2_edges, 'clust_coeff1':clust1_edges, 'clust_coeff2':clust2_edges, 'ave_neigh_deg1':ave_neigh_deg1_edges, 'ave_neigh_deg2':ave_neigh_deg2_edges,\
         'eig_cent1':eig_cent1_edges, 'eig_cent2':eig_cent2_edges, 'deg_cent1':deg_cent1_edges, 'deg_cent2':deg_cent2_edges, 'clos_cent1':closn_cent1_edges, 'clos_cent2':closn_cent2_edges, 'betw_cent1':betw_cent1_edges, 'betw_cent2':betw_cent2_edges, \
         'load_cent1':load_cent1_edges, 'load_cent2':load_cent2_edges, 'ktz_cent1':ktz_cent1_edges, 'ktz_cent2':ktz_cent2_edges, 'pref_attach':df_pref_attach, 'LHN':LHN_edges, 'svd_edges':svd_edges,'svd_edges_dot':svd_edges_dot,'svd_edges_mean':svd_edges_mean,\
         'svd_edges_approx':svd_edges_approx,'svd_edges_dot_approx':svd_edges_dot_approx,'svd_edges_mean_approx':svd_edges_mean_approx, 'short_path':short_path_edges, 'deg_assort':deg_ass_net, 'transit_net':transit_net, 'diam_net':diam_net, \
         'num_nodes':num_nodes, 'num_edges':num_edges}     
    
    # construct a dataframe of the features
    df_feat = pd.DataFrame(data=d)
    df_feat['ind'] = df_feat.index
    df_feat = pd.merge(df_feat,df_merge, on=['ind','i','j'], sort=False)
    return df_feat


def creat_full_set(df_t,df_f):
    
    """ 
    This reads dataframes created for positive and negative class, join them with their associated label.

    Input and Parameters:
    -------
    df_t: datafram of features for true edges
    df_f: datafram of features for true non-edges

    Returns
    -------
    df_all: a data frames with columns of features and ground truth 

    Examples:
    -------
    df_all = creat_full_set(df_t,df_f)
    """

    df_t = df_t.drop_duplicates(subset=['i','j'], keep="first")
    df_f = df_f.drop_duplicates(subset=['i','j'], keep="first")

    df_t.insert(2, "TP", 1, True)
    df_f.insert(2, "TP", 0, True)
    
    df_all = [df_t, df_f]
    df_all = pd.concat(df_all)
    
    # data cleaning
    df_all.loc[df_all['short_path'] == np.inf,'short_path'] = 1000*max(df_all.loc[~(df_all['short_path'] == np.inf),'short_path'])
    df_all.loc[df_all['diam_net'] == np.inf,'diam_net'] = 1e6
     
    return df_all


def creat_numpy_files(dir_results, df_ho, df_tr):
    
    """ 
    This function reads dataframes created for positive and negative classes, join them with their associated label.

    Input and Parameters:
    -------
    df_tr: datafram of features/ground truth for positive and negative class for model selection
    df_ho: datafram of features/ground truth for positive and negative class for held out model performance

    Returns:
    -------
    save numpy files of X_train_i and y_train_i for 5 folds, also X_Eseen/X_Eunseen, y_Eseen/y_Eunseen in dir_results

    Example:
    -------
    creat_numpy_files(dir_results, df_ho, df_tr)
    """
    
    feature_set = ['com_ne', 'ave_deg_net', 'var_deg_net', 'ave_clust_net',
           'num_triangles_1', 'num_triangles_2', 'page_rank_pers_edges',
           'pag_rank1', 'pag_rank2', 'clust_coeff1', 'clust_coeff2',
           'ave_neigh_deg1', 'ave_neigh_deg2', 'eig_cent1', 'eig_cent2',
           'deg_cent1', 'deg_cent2', 'clos_cent1', 'clos_cent2', 'betw_cent1',
           'betw_cent2', 'load_cent1', 'load_cent2', 'ktz_cent1', 'ktz_cent2',
           'pref_attach', 'LHN', 'svd_edges', 'svd_edges_dot', 'svd_edges_mean',
           'svd_edges_approx', 'svd_edges_dot_approx', 'svd_edges_mean_approx',
           'short_path', 'deg_assort', 'transit_net', 'diam_net',
           'jacc_coeff', 'res_alloc_ind', 'adam_adar' , 'num_nodes','num_edges']  

    X_test_heldout = df_ho
    y_test_heldout = np.array(df_ho.TP)
    
    
    X_train_orig = df_tr
    y_train_orig = np.array(df_tr.TP)

    skf = StratifiedKFold(n_splits=5,shuffle=True)
    skf.get_n_splits(X_train_orig, y_train_orig)

    if not os.path.isdir(dir_results+'/'):
        os.mkdir(dir_results+'/')
        
    nFold = 1 
    for train_index, test_index in skf.split(X_train_orig, y_train_orig):

        cv_train = list(train_index)
        cv_test = list(test_index)
         
         
        train = X_train_orig.iloc[np.array(cv_train)]
        test = X_train_orig.iloc[np.array(cv_test)]

        y_train = train.TP
        y_test = test.TP
        

        X_train = train.loc[:,feature_set]
        X_test = test.loc[:,feature_set]

        X_test.fillna(X_test.mean(), inplace=True)
        X_train.fillna(X_train.mean(), inplace=True)

        sm = RandomOverSampler(random_state=42)
        X_train, y_train = sm.fit_sample(X_train, y_train)

        np.save(dir_results+'/X_trainE_'+'cv'+str(nFold), X_train)
        np.save(dir_results+'/y_trainE_'+'cv'+str(nFold), y_train)
        np.save(dir_results+'/X_testE_'+'cv'+str(nFold), X_test)
        np.save(dir_results+'/y_testE_'+'cv'+str(nFold), y_test)

        print( "created fold ",nFold, " ...")
        
        nFold = nFold + 1

    seen = X_train_orig
    y_seen = seen.TP
    X_seen = seen.loc[:,feature_set]
    X_seen.fillna(X_seen.mean(), inplace=True)  

    # balance train set with upsampling
    sm = RandomOverSampler(random_state=42)
    X_seen, y_seen = sm.fit_sample(X_seen, y_seen)

    np.save(dir_results+'/X_Eseen', X_seen)
    np.save(dir_results+'/y_Eseen', y_seen)
    print( "created train set ...")


    unseen = X_test_heldout
    y_unseen = unseen.TP
    X_unseen = unseen.loc[:,feature_set]
    X_unseen.fillna(X_unseen.mean(), inplace=True) 

    np.save(dir_results+'/X_Eunseen', X_unseen)
    np.save(dir_results+'/y_Eunseen', y_unseen) 
    print( "created holdout set ...")
    
    
def model_selection(path_to_data, path_to_results, n_depths, n_ests):
    
    """ 
    This function runs cross validation on train set and finds the random forest model parameters which yeilds to best fmeasure.

    Input and Parameters:
    -------
    path_to_data: path to held out featute matrices 
    path_to_results: path to save model performance ast txt file
    n_depth: a list of max_depths for random forest parameter
    n_est: a list of n_estimators for random forest parameter

    Returns:
    -------
    n_depth: n_depth which yeild to maximum fmeasure
    n_est: n_est which yeild to maximum fmeasure

    Examples:
    -------
    n_depth, ne_est = model_selection(path_to_data, path_to_results, n_depths, n_ests)
    """
    
    fmeasure_matrix = np.zeros((len(n_depths),len(n_ests)))
    
    if not os.path.isdir(path_to_results):
        os.mkdir(path_to_results)
    
    # load train and validation set for each fold
    X_train = {}
    y_train = {}
    X_test = {}
    y_test = {}
    for nFold in range(1,6):
        
        exec("X_train["+ str(nFold) +"] = np.load( path_to_data + '/X_trainE_cv"+ str(nFold) +".npy')")
        exec("y_train["+ str(nFold) +"] = np.load( path_to_data + '/y_trainE_cv"+ str(nFold) +".npy')")
        exec("X_test["+ str(nFold) +"] = np.load( path_to_data + '/X_testE_cv"+ str(nFold) +".npy')")
        exec("y_test["+ str(nFold) +"] = np.load( path_to_data + '/y_testE_cv"+ str(nFold) +".npy')")
    
    # run a grid search for parameter tuning 
    print("start grid search ... ")
    for n_ii, ii in enumerate(n_depths):
        for n_jj, jj in enumerate(n_ests):
        
            auc_measure = []     
            precision_total = np.zeros((5,2))
            recall_total = np.zeros((5,2))
            f_measure_total = np.zeros((5,2))
            
            for cv in range(1,6):
                
                 Xtr = X_train[cv]
                 ytr = y_train[cv]
                 Xts = X_test[cv]
                 yts = y_test[cv]
                
                 # train the model 
                 
                 dtree_model = RandomForestClassifier(max_depth=ii,n_estimators=jj).fit(Xtr, ytr)
                    
                 # predict for test test
                 dtree_predictions = dtree_model.predict(Xts)
                 dtree_proba = dtree_model.predict_proba(Xts)
                        
                 # calculate performance metrics
                 cm_dt4 = confusion_matrix(yts, dtree_predictions)
                 
                 auc_aux = roc_auc_score(yts, dtree_proba[:,1])
                 auc_measure.append(auc_aux)
                 
                 precision_aux, recall_aux, f_measure_aux, _ = precision_recall_fscore_support(yts, dtree_predictions, average=None)
                 precision_total[cv-1,:] = precision_aux
                 recall_total[cv-1,:] = recall_aux
                 f_measure_total[cv-1,:] = f_measure_aux
              
            # take average of performance metrics across folds
            mean_auc = np.mean(auc_measure)
            mean_precision = np.mean(precision_total,axis=0)
            mean_recall = np.mean(recall_total,axis=0)
            mean_f_measure = np.mean(f_measure_total,axis=0)
            
            # write the result in text file
            f = open( path_to_results + '/RF_Best_metrics.txt','w')
            f.write('mean_AUC = '+ str(mean_auc)+'\n')
            f.write('mean_precision = '+ str(mean_precision)+'\n')
            f.write('mean_recall = '+ str(mean_recall)+'\n')
            f.write('mean_f_measure = '+ str(mean_f_measure)+'\n')            
            f.close()
            
            # keep track of average fmeasure for each parameter set
            
            fmeasure_matrix[n_ii,n_jj] = mean_f_measure[0]
            
    # find the model parameters which gives the best average fmeasure on 5 fold validation sets    
    i,j = np.unravel_index(fmeasure_matrix.argmax(), fmeasure_matrix.shape)
    n_depth = n_depths[i]
    ne_est = n_ests[j]
    print("best parameters for random forest are: n_depth: "+str(n_depth)+", and n_estimators: "+str(ne_est))
    return n_depth, ne_est
        
        
def heldout_performance(path_to_data, path_to_results, n_depth, n_est):
    
    """ 
    This function trains a random forest model on seen data and performs prediction on heldout.

    Input and Parameters:
    -------
    path_to_data: path to held out featute matrices 
    path_to_results: path to save model performance ast txt file
    n_depth: max_depth for random forest parameter
    n_est: n_estimators for random forest parameter

    Returns:
    -------
    auc_measure: auc on heldout
    precision_total: precision of positive class on heldout
    recall_total: recall of positive class on heldout

    Examples:
    -------
    auc , precision, recall = heldout_performance(path_to_data, path_to_results, n_depth, n_est)
    """
    
    if not os.path.isdir(path_to_results):
        os.mkdir(path_to_results)
    f = open(path_to_results + '/RF_Best_metrics.txt','w')
    path_to_data = './feature_metrices'
    
    # read data
    X_train = np.load(path_to_data+'/X_Eseen.npy')
    y_train = np.load(path_to_data+'/y_Eseen.npy')
    X_test = np.load(path_to_data+'/X_Eunseen.npy')
    y_test = np.load(path_to_data+'/y_Eunseen.npy')
    
    
    col_mean = np.nanmean(X_train, axis=0)
    inds = np.where(np.isnan(X_train))
    X_train[inds] = np.take(col_mean, inds[1])
    
    col_mean = np.nanmean(X_test, axis=0)
    inds = np.where(np.isnan(X_test))
    X_test[inds] = np.take(col_mean, inds[1])
     
       
    # train the model
    dtree_model = RandomForestClassifier(n_estimators=n_est,max_depth=n_depth).fit(X_train, y_train)
    
    
    # feature importance and prediction on test set 
    feature_importance = dtree_model.feature_importances_
    dtree_predictions = dtree_model.predict(X_test)
    dtree_proba = dtree_model.predict_proba(X_test)
      
    # calculate performance metrics
    cm_dt4 = confusion_matrix(y_test, dtree_predictions)
    auc_measure = roc_auc_score(y_test, dtree_proba[:,1])
       
     
    precision_total, recall_total, f_measure_total, _ = precision_recall_fscore_support(y_test, dtree_predictions, average=None)
       
    
    
    f.write('heldout_AUC = '+ str(auc_measure)+'\n')
    f.write('heldout_precision = '+ str(precision_total)+'\n')
    f.write('heldout_recall = '+ str(recall_total)+'\n')
    f.write('heldout_f_measure = '+ str(f_measure_total)+'\n')
    f.write('feature_importance = '+ str(list(feature_importance))+'\n')
    f.close()
    
    print("AUC: " +str(np.round(auc_measure,2)))
    print("precision: " +str(np.round(precision_total[0],2)))
    print("recall: " +str(np.round(recall_total[0],2)))
    return auc_measure, precision_total[0], recall_total[0]


def demo(): 
    
    """ 
    This function extracts topological features and performs link prediction using stacking model on a sample network.

    Input and Parameters:
    -------

    Returns:
    -------

    Examples:
    -------
    >>> demo()
    """
    
    #### load the original netowrk A_orig
    path_E_orig = "./edge_orig.txt"
    edges_orig = np.loadtxt(path_E_orig,comments = '#')
    edges_orig = np.array(np.matrix(edges_orig))
    num_nodes = int(np.max(edges_orig)) + 1
    row = np.array(edges_orig)[:,0]
    col = np.array(edges_orig)[:,1]

    data_aux = np.ones(len(row))
    A_orig = csr_matrix((data_aux,(row,col)),shape=(num_nodes,num_nodes))
    A_orig = sparse.triu(A_orig,1) + sparse.triu(A_orig,1).transpose()
    A_orig[A_orig>0] = 1 
    A_orig = A_orig.todense()

    #### construct the holdout and training matriced from the original matrix
    alpha = 0.8 # sampling rate for holdout matrix
    alpha_ = 0.8 # sampling rate for training matrix
    A_ho, A_tr = gen_tr_ho_networks(A_orig, alpha, alpha_)
    
    #### extract features #####
    sample_true_false_edges(A_orig, A_tr, A_ho)
    edge_t_tr = np.loadtxt("./edge_tf_tr/edge_t.txt").astype('int')
    edge_f_tr = np.loadtxt("./edge_tf_tr/edge_f.txt").astype('int')
    df_f_tr = gen_topol_feats(A_orig, A_tr, edge_f_tr)
    df_t_tr = gen_topol_feats(A_orig, A_tr, edge_t_tr)
    
    edge_t_ho = np.loadtxt("./edge_tf_ho/edge_t.txt").astype('int')
    edge_f_ho = np.loadtxt("./edge_tf_ho/edge_f.txt").astype('int')
    df_f_ho = gen_topol_feats(A_orig, A_ho, edge_f_ho)
    df_t_ho = gen_topol_feats(A_orig, A_ho, edge_t_ho)
    
    feat_path = "./ef_gen_tr/"
    if not os.path.isdir(feat_path):
        os.mkdir(feat_path)
    df_t_tr.to_pickle(feat_path + 'df_t')
    df_f_tr.to_pickle(feat_path + 'df_f')
    
    feat_path = "./ef_gen_ho/"
    if not os.path.isdir(feat_path):
        os.mkdir(feat_path)
    df_t_ho.to_pickle(feat_path + 'df_t')
    df_f_ho.to_pickle(feat_path + 'df_f')
    
    
    #### load dataframes for train and holdout sets ####
    df_tr = creat_full_set(df_t_tr,df_f_tr)
    df_ho = creat_full_set(df_t_ho,df_f_ho)
    
    #### creat and save feature matrices #### 
    dir_output = './feature_metrices'  # output path
    creat_numpy_files(dir_output, df_ho, df_tr)
    
    
    #### perform model selection #### 
    path_to_data = './feature_metrices' 
    path_to_results = './results'
    n_depths = [3, 6] # here is a sample search space
    n_ests = [25, 50, 100] # here is a sample search space
    n_depth, n_est = model_selection(path_to_data, path_to_results, n_depths, n_ests)
    
    
    #### perform model selection #### 
    auc , precision, recall = heldout_performance(path_to_data, path_to_results, n_depth, n_est)
   

def topol_stacking(edges_orig): 
    
    """ 
    This function extracts topological features and performs link prediction using stacking model on the input network (edges_orig).

    Input and Parameters:
    -------
    edges_orig: the original edge list

    Returns:
    -------

    Examples:
    -------
    >>> topol_stacking(edges_orig)
    """
    
    #### load the original netowrk A_orig
    num_nodes = int(np.max(edges_orig)) + 1
    row = np.array(edges_orig)[:,0]
    col = np.array(edges_orig)[:,1]

    data_aux = np.ones(len(row))
    A_orig = csr_matrix((data_aux,(row,col)),shape=(num_nodes,num_nodes))
    A_orig = sparse.triu(A_orig,1) + sparse.triu(A_orig,1).transpose()
    A_orig[A_orig>0] = 1 
    A_orig = A_orig.todense()

    #### construct the holdout and training matriced from the original matrix
    alpha = 0.8 # sampling rate for holdout matrix
    alpha_ = 0.8 # sampling rate for training matrix
    A_ho, A_tr = gen_tr_ho_networks(A_orig, alpha, alpha_)
    
    #### extract features #####
    sample_true_false_edges(A_orig, A_tr, A_ho)
    edge_t_tr = np.loadtxt("./edge_tf_tr/edge_t.txt").astype('int')
    edge_f_tr = np.loadtxt("./edge_tf_tr/edge_f.txt").astype('int')
    df_f_tr = gen_topol_feats(A_orig, A_tr, edge_f_tr)
    df_t_tr = gen_topol_feats(A_orig, A_tr, edge_t_tr)
    
    edge_t_ho = np.loadtxt("./edge_tf_ho/edge_t.txt").astype('int')
    edge_f_ho = np.loadtxt("./edge_tf_ho/edge_f.txt").astype('int')
    df_f_ho = gen_topol_feats(A_orig, A_ho, edge_f_ho)
    df_t_ho = gen_topol_feats(A_orig, A_ho, edge_t_ho)
    
    feat_path = "./ef_gen_tr/"
    if not os.path.isdir(feat_path):
        os.mkdir(feat_path)
    df_t_tr.to_pickle(feat_path + 'df_t')
    df_f_tr.to_pickle(feat_path + 'df_f')
    
    feat_path = "./ef_gen_ho/"
    if not os.path.isdir(feat_path):
        os.mkdir(feat_path)
    df_t_ho.to_pickle(feat_path + 'df_t')
    df_f_ho.to_pickle(feat_path + 'df_f')
    
    
    #### load dataframes for train and holdout sets ####
    df_tr = creat_full_set(df_t_tr,df_f_tr)
    df_ho = creat_full_set(df_t_ho,df_f_ho)
    
    #### creat and save feature matrices #### 
    dir_output = './feature_metrices'  # output path
    creat_numpy_files(dir_output, df_ho, df_tr)
    
    
    #### perform model selection #### 
    path_to_data = './feature_metrices' 
    path_to_results = './results'
    n_depths = [3, 6] # here is a sample search space
    n_ests = [25, 50, 100] # here is a sample search space
    n_depth, n_est = model_selection(path_to_data, path_to_results, n_depths, n_ests)
    
    
    #### perform model selection #### 
    auc , precision, recall = heldout_performance(path_to_data, path_to_results, n_depth, n_est)
   
