import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.cluster import KMeans
import gseapy
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

# for rpy2
base = importr('base')

def overlap_coefficient(list1, list2):
    # Szymkiewicz–Simpson coefficient
    intersection = len(list(set(list1).intersection(list(set(list2)))))
    smaller_set = min(len(list1), len(list2))
    return float(intersection) / smaller_set

def SVD_scores(mat, paths):
    # Create pathway matrices
    # Transpose the matrix for SVD

#     mat.loc[:, :] = StandardScaler().fit_transform(mat.to_numpy())
    mat_t = mat.T
    pathway_activities = []

    for k, v in paths.items():
        pathway_mat = mat_t.iloc[mat_t.index.isin(v), :]
        pathway_mat = pathway_mat.to_numpy(dtype=float)

        # s = singular values
        # u = left singular vector
        # v = right singular vector
        u, s, vh = np.linalg.svd(pathway_mat)

        pathway_activities.append(vh[0])

    pathway_activities_df = pd.DataFrame(pathway_activities, columns=mat.index, index=paths.keys())
    return pathway_activities_df


def t_tests(matrix, classes, multiple_correction_method, testtype="ttest"):
    metabolites = matrix.columns.tolist()
    matrix['Target'] = pd.factorize(classes)[0]
    disease = matrix.loc[matrix["Target"] == 0]
    disease.drop(['Target'], axis=1, inplace=True)
    ctrl = matrix.loc[matrix["Target"] != 0]
    ctrl.drop(['Target'], axis=1, inplace=True)
    if testtype == "mwu":
        pvalues = stats.mannwhitneyu(disease, ctrl, axis=0)[1]
    else:
        pvalues = stats.ttest_ind(disease, ctrl)[1]
        
    padj = sm.stats.multipletests(pvalues, 0.05, method=multiple_correction_method)
    results = pd.DataFrame(zip(metabolites, pvalues, padj[1]),
                           columns=["Metabolite", "P-value", "P-adjust"])
    return results

def over_representation_analysis(DA_list, background_list, pathway_dict):
    """
    Function for over representation analysis using Fisher exact test (right tailed)
    :param DA_list: List of differentially abundant metabolite IDENTIFIERS
    :param background_list: background list of IDENTIFIERS
    :param pathways_df: pathway dataframe containing compound identifiers
    :return: DataFrame of ORA results for each pathway, p-value, q-value, hits ratio
    """

    pathways = pathway_dict.keys()
    pathway_names = pathway_dict.keys()

    pathways_with_compounds = []
    pathway_names_with_compounds = []
    pvalues = []
    pathway_ratio = []
    pathway_count = 0
    pathway_coverage = []

    for pathway in pathways:
        # perform ORA for each pathway
        pathway_compounds = pathway_dict[pathway]
        pathway_compounds = [i for i in pathway_compounds if str(i) != "nan"]
        if not pathway_compounds or len(pathway_compounds) < 2:
            # ignore pathway if contains no compounds or has less than 3 compounds
            continue
        else:

            DA_in_pathway = len(set(DA_list) & set(pathway_compounds))
            # k: compounds in DA list AND pathway
            DA_not_in_pathway = len(np.setdiff1d(DA_list, pathway_compounds))
            # K: compounds in DA list not in pathway
            compound_in_pathway_not_DA = len(set(pathway_compounds) & set(np.setdiff1d(background_list, DA_list)))
            # not DEM compounds present in pathway
            compound_not_in_pathway_not_DA = len(
                np.setdiff1d(np.setdiff1d(background_list, DA_list), pathway_compounds))
            # compounds in background list not present in pathway
            if DA_in_pathway == 0 or (compound_in_pathway_not_DA + DA_in_pathway) < 2:
                # ignore pathway if there are no DEM compounds in that pathway
                continue
            else:
                pathway_count += 1
                # Create 2 by 2 contingency table
                pathway_ratio.append(str(DA_in_pathway) + "/" + str(compound_in_pathway_not_DA + DA_in_pathway))
                pathway_coverage.append(
                    str(compound_in_pathway_not_DA + DA_in_pathway) + "/" + str(len(pathway_compounds)))
                pathways_with_compounds.append(pathway)
                pathway_names_with_compounds.append(pathway)
                contingency_table = np.array([[DA_in_pathway, compound_in_pathway_not_DA],
                                              [DA_not_in_pathway, compound_not_in_pathway_not_DA]])
                # Run right tailed Fisher's exact test
                oddsratio, pvalue = stats.fisher_exact(contingency_table, alternative="greater")
                pvalues.append(pvalue)
    try:
        padj = sm.stats.multipletests(pvalues, 0.05, method="fdr_bh")
        results = pd.DataFrame(
            zip(pathways_with_compounds, pathway_names_with_compounds, pathway_ratio, pathway_coverage, pvalues,
                padj[1]),
            columns=["Pathway_ID", "Pathway_name", "Hits", "Coverage", "P-value", "P-adjust"])
    except ZeroDivisionError:
        padj = [1] * len(pvalues)
        results = pd.DataFrame(zip(pathways_with_compounds, pathway_names_with_compounds, pathway_ratio, pvalues, padj),
                               columns=["Pathway_ID", "Pathway_name", "Hits", "Coverage", "P-value", "P-adjust"])
    return results


def svd_score_ttest(mat, pathways, testtype):
    pathway_scores = SVD_scores(mat.iloc[:, :-1], pathways)
    t_test_paths = t_tests(pathway_scores.T, mat["Group"], "fdr_bh", testtype)
    sig_paths_05 = t_test_paths[t_test_paths["P-adjust"] <= 0.05]
    t_test_paths = t_test_paths.sort_values(by="P-value").reset_index()
    return t_test_paths


def ora_results(mat, pathways):
    t_test_res = t_tests(mat.iloc[:, :-1], mat["Group"], "fdr_bh")
    DA_compounds = t_test_res[t_test_res["P-adjust"] < 0.05]["Metabolite"].tolist()
    bg_set = mat.iloc[:, :-1].columns.to_list()
    ora_res = over_representation_analysis(DA_compounds, bg_set, pathways)
    try:
        ora_res = ora_res[~ora_res["Pathway_ID"].isin(["R-HSA-1430728", "R-HSA-1643685", "R-HSA-382551"])]
    except:
        pass
    ora_res = ora_res.rename(columns={"Pathway_ID": "ID"})
    return ora_res


def gsea_results(mat, gmt):
    gsea_res = gseapy.gsea(data=mat.iloc[:, :-1].T,
                           gene_sets=gmt,
                           cls=mat["Group"],
                           method='signal_to_noise',
                           min_size=2,
                           max_size=2000,
                           verbose=False,
                           outdir=None)
    gsea_res_all = gsea_res.res2d.sort_values(by="pval")
    
    try:
        gsea_res_all = gsea_res_all.drop(["R-HSA-1430728", "R-HSA-1643685", "R-HSA-382551"], axis=0)
    except:
        pass
    gsea_res_all = gsea_res_all.reset_index()
    gsea_res_all = gsea_res_all.rename(columns={"Term": "ID"})
    return gsea_res_all


def ssgsea_results(mat, pathways, testtype):
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_mat = ro.conversion.py2rpy(mat.iloc[:, :-1].T)
    r_mat = base.as_matrix(r_mat)  # abundance matrix
    row_vec = base.as_character(mat.columns[:-1].tolist())
    r_mat.rownames = row_vec
    r_list = ro.ListVector(pathways)  # pathways
    gsva_r = importr('GSVA')
    gsva_res = gsva_r.gsva(r_mat, r_list, method="ssgsea")
    with localconverter(ro.default_converter + pandas2ri.converter):
        gsva_df = ro.conversion.rpy2py(gsva_res)
    ssgsea_scores = pd.DataFrame(gsva_df, index=pathways.keys(), columns=mat.iloc[:, :-1].index.tolist())
    t_test_ssgsea = t_tests(ssgsea_scores.T, mat["Group"], "fdr_bh", testtype)
    t_test_ssgsea = t_test_ssgsea.sort_values(by="P-value").reset_index()
    return t_test_ssgsea


def gsva_ttest(mat, pathways, testtype):
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_mat = ro.conversion.py2rpy(mat.iloc[:, :-1].T)
    r_mat = base.as_matrix(r_mat)  # abundance matrix
    row_vec = base.as_character(mat.columns[:-1].tolist())
    r_mat.rownames = row_vec
    r_list = ro.ListVector(pathways)  # pathways
    gsva_r = importr('GSVA')
    gsva_res = gsva_r.gsva(r_mat, r_list)
    with localconverter(ro.default_converter + pandas2ri.converter):
        gsva_df = ro.conversion.rpy2py(gsva_res)
    gsva_res_df = pd.DataFrame(gsva_df, index=pathways.keys(), columns=mat.iloc[:, :-1].index.tolist())
    t_test_gsva = t_tests(gsva_res_df.T, mat["Group"], "fdr_bh", testtype)
    t_test_gsva = t_test_gsva.sort_values(by="P-value").reset_index()
    return t_test_gsva


def svd_classifier(mat, pathways):
    pathway_scores = SVD_scores(mat.iloc[:, :-1], pathways)
    labels_binary = pd.factorize(mat["Group"])[0]
    pls2 = PLSRegression(n_components=5).fit(pathway_scores.T, labels_binary)
    coefs = list(zip([i[0] for i in pls2.coef_], pathway_scores.T.columns))
    coefs_df = pd.DataFrame(coefs, columns=["Coef", "ID"])
    coefs_df = coefs_df.sort_values(by="Coef", ascending=False, key=abs).reset_index()
    coefs_df["Coef"] = coefs_df["Coef"].abs()
    return coefs_df


def ssgsea_classifier_old(mat, pathways):
    ssgsea_expr = mat.iloc[:, :-1].T
    ssgsea_mat = gseapy.ssgsea(data=ssgsea_expr,
                               gene_sets="../pathway_databases/Reactome_human_pathways_compounds_R76.gmt",
                               no_plot=True,
                               min_size=2,
                               outdir="/tmpoutdir/")
    ssgsea_scores = ssgsea_mat.res2d
    ssgsea_scores = ssgsea_scores.drop(["R-HSA-1430728", "R-HSA-1643685", "R-HSA-382551"], axis=0)
    labels_binary = pd.factorize(mat["Group"])[0]
    pls2 = PLSRegression(n_components=5).fit(ssgsea_scores.T, labels_binary)
    coefs = list(zip([i[0] for i in pls2.coef_], ssgsea_scores.T.columns))
    coefs_df = pd.DataFrame(coefs, columns=["Coef", "ID"])
    coefs_df = coefs_df.sort_values(by="Coef", ascending=False, key=abs).reset_index()
    coefs_df["Coef"] = coefs_df["Coef"].abs()
    return coefs_df

def ssgsea_classifier(mat, pathways):
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_mat = ro.conversion.py2rpy(mat.iloc[:, :-1].T)
    r_mat = base.as_matrix(r_mat)  # abundance matrix
    r_list = ro.ListVector(pathways)  # pathways
    gsva_r = importr('GSVA')
    gsva_res = gsva_r.gsva(r_mat, r_list, method="ssgsea")
    with localconverter(ro.default_converter + pandas2ri.converter):
        gsva_df = ro.conversion.rpy2py(gsva_res)
    gsva_res_df = pd.DataFrame(gsva_df, index=pathways.keys(), columns=mat.iloc[:, :-1].index.tolist())
    labels_binary = pd.factorize(mat["Group"])[0]
    pls2 = PLSRegression(n_components=5).fit(gsva_res_df.T, labels_binary)
    coefs = list(zip([i[0] for i in pls2.coef_], gsva_res_df.T.columns))
    coefs_df = pd.DataFrame(coefs, columns=["Coef", "ID"])
    coefs_df = coefs_df.sort_values(by="Coef", ascending=False, key=abs).reset_index()
    coefs_df["Coef"] = coefs_df["Coef"].abs()
    return coefs_df


def gsva_classifier(mat, pathways):
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_mat = ro.conversion.py2rpy(mat.iloc[:, :-1].T)
    r_mat = base.as_matrix(r_mat)  # abundance matrix
    r_list = ro.ListVector(pathways)  # pathways
    gsva_r = importr('GSVA')
    gsva_res = gsva_r.gsva(r_mat, r_list)
    with localconverter(ro.default_converter + pandas2ri.converter):
        gsva_df = ro.conversion.rpy2py(gsva_res)
    gsva_res_df = pd.DataFrame(gsva_df, index=pathways.keys(), columns=mat.iloc[:, :-1].index.tolist())
    labels_binary = pd.factorize(mat["Group"])[0]
    pls2 = PLSRegression(n_components=5).fit(gsva_res_df.T, labels_binary)
    coefs = list(zip([i[0] for i in pls2.coef_], gsva_res_df.T.columns))
    coefs_df = pd.DataFrame(coefs, columns=["Coef", "ID"])
    coefs_df = coefs_df.sort_values(by="Coef", ascending=False, key=abs).reset_index()
    coefs_df["Coef"] = coefs_df["Coef"].abs()
    return coefs_df


def zscore_classifier(mat, paths):
#     mat.loc[:, :] = StandardScaler().fit_transform(mat.to_numpy())
    mat_t = mat.T
    pathway_activities = []

    for k, v in paths.items():
        pathway_mat = mat_t.iloc[mat_t.index.isin(v), :]
        pathway_mat = pathway_mat.to_numpy(dtype=float)
        zscores = stats.zscore(pathway_mat, axis=1)
        # avg_zscore = np.mean(zscores, axis=0)
        # pathway_act = avg_zscore / np.sqrt(pathway_mat.shape[0])
        sum_zscore = np.sum(zscores, axis=0)
        pathway_act = sum_zscore / np.sqrt(pathway_mat.shape[0])
        pathway_activities.append(pathway_act)

    pathway_activities_df = pd.DataFrame(pathway_activities, columns=mat.index, index=paths.keys())
    labels_binary = pd.factorize(mat["Group"])[0]
    pls2 = PLSRegression(n_components=5).fit(pathway_activities_df.T, labels_binary)
    coefs = list(zip([i[0] for i in pls2.coef_], pathway_activities_df.T.columns))
    coefs_df = pd.DataFrame(coefs, columns=["Coef", "ID"])
    coefs_df = coefs_df.sort_values(by="Coef", ascending=False, key=abs).reset_index()
    coefs_df["Coef"] = coefs_df["Coef"].abs()
    return coefs_df

def zscore_res(mat, pathways, testtype):
#     mat.loc[:, :-1] = StandardScaler().fit_transform(mat.iloc[:, :-1].to_numpy())
    mat_t = mat.T
    pathway_activities = []

    for k, v in pathways.items():
        pathway_mat = mat_t.iloc[mat_t.index.isin(v), :]
        pathway_mat = pathway_mat.to_numpy(dtype=float)
        zscores = stats.zscore(pathway_mat, axis=1)
        # avg_zscore = np.mean(zscores, axis=0)
        # pathway_act = avg_zscore / np.sqrt(pathway_mat.shape[0])
        sum_zscore = np.sum(zscores, axis=0)
        pathway_act = sum_zscore / np.sqrt(pathway_mat.shape[0])
        pathway_activities.append(pathway_act)
    pathway_activities_df = pd.DataFrame(pathway_activities, columns=mat.index, index=pathways.keys())
    t_test_res = t_tests(pathway_activities_df.T, mat["Group"], "fdr_bh", testtype)
    return t_test_res


def cluster_classifier(mat, pathways):
    """
    Pathway activity scores based on k-means clustering distance to the first centroid. PLS classifier applied.
    :param mat:
    :param pathways:
    :return:
    """
    pathway_matrices = []
    pathway_ids = []
    for pathway, compounds in pathways.items():
        single_pathway_matrix = mat.drop(mat.columns.difference(compounds), axis=1)
        if single_pathway_matrix.shape[1] >= 1:
            pathway_matrices.append(single_pathway_matrix.values)
            pathway_ids.append(pathway)

    scores = []
    for m in pathway_matrices:
        kmeans = KMeans(n_clusters=2)
        new_data = kmeans.fit_transform(m)
        scores.append(new_data[:, 0])
    scores_df = pd.DataFrame(scores, columns=mat.index, index=pathways.keys())
    labels_binary = pd.factorize(mat["Group"])[0]
    pls2 = PLSRegression(n_components=5).fit(scores_df.T, labels_binary)
    coefs = list(zip([i[0] for i in pls2.coef_], scores_df.T.columns))
    coefs_df = pd.DataFrame(coefs, columns=["Coef", "ID"])
    coefs_df = coefs_df.sort_values(by="Coef", ascending=False, key=abs).reset_index()
    coefs_df["Coef"] = coefs_df["Coef"].abs()
    return coefs_df

def cluster_ttest(mat, pathways, testtype):
#     mat.loc[:, :-1] = StandardScaler().fit_transform(mat.iloc[:, :-1].to_numpy())
    pathway_matrices = []
    pathway_ids = []
    for pathway, compounds in pathways.items():
        single_pathway_matrix = mat.drop(mat.columns.difference(compounds), axis=1)
        if single_pathway_matrix.shape[1] >= 1:
            pathway_matrices.append(single_pathway_matrix.values)
            pathway_ids.append(pathway)

    scores = []
    for m in pathway_matrices:
        kmeans = KMeans(n_clusters=2)
        new_data = kmeans.fit_transform(m)
        scores.append(new_data[:, 0])
    scores_df = pd.DataFrame(scores, columns=mat.index, index=pathways.keys())
    t_test_res = t_tests(scores_df.T, mat["Group"], "fdr_bh", testtype)
    return t_test_res

def cluster_coefs_projection(mat, pathways):
    pathway_matrices = []
    pathway_ids = []
    for pathway, compounds in pathways.items():
        single_pathway_matrix = mat.drop(mat.columns.difference(compounds), axis=1)
        if single_pathway_matrix.shape[1] >= 1:
            pathway_matrices.append(single_pathway_matrix.values)
            pathway_ids.append(pathway)

    scores = []
    for m in pathway_matrices:
        kmeans = KMeans(n_clusters=2).fit(m)
        centroids1 = kmeans.cluster_centers_[0]
        centroids2 = kmeans.cluster_centers_[1]

        vec = centroids1 - centroids2
        unit_vec = vec / np.linalg.norm(vec)
        proj_data = unit_vec.dot(m.T)
        scores.append(proj_data)

    scores_df = pd.DataFrame(scores, columns=mat.index, index=pathways.keys())
    labels_binary = pd.factorize(mat["Group"])[0]
    pls2 = PLSRegression(n_components=5).fit(scores_df.T, labels_binary)
    coefs = list(zip([i[0] for i in pls2.coef_], scores_df.T.columns))
    coefs_df = pd.DataFrame(coefs, columns=["Coef", "ID"])
    coefs_df = coefs_df.sort_values(by="Coef", ascending=False, key=abs).reset_index()
    coefs_df["Coef"] = coefs_df["Coef"].abs()
    return coefs_df

def clusterproj_ttest(mat, pathways, testtype):
#     mat.loc[:, :-1] = StandardScaler().fit_transform(mat.iloc[:, :-1].to_numpy())
    pathway_matrices = []
    pathway_ids = []
    for pathway, compounds in pathways.items():
        single_pathway_matrix = mat.drop(mat.columns.difference(compounds), axis=1)
        if single_pathway_matrix.shape[1] >= 1:
            pathway_matrices.append(single_pathway_matrix.values)
            pathway_ids.append(pathway)

    scores = []
    for m in pathway_matrices:
        kmeans = KMeans(n_clusters=2).fit(m)
        centroids1 = kmeans.cluster_centers_[0]
        centroids2 = kmeans.cluster_centers_[1]

        vec = centroids1 - centroids2
        unit_vec = vec / np.linalg.norm(vec)
        proj_data = unit_vec.dot(m.T)
        scores.append(proj_data)

    scores_df = pd.DataFrame(scores, columns=mat.index, index=pathways.keys())
    t_test_res = t_tests(scores_df.T, mat["Group"], "fdr_bh", testtype)
    return t_test_res

def kernel_pca_classifier(mat, pathways):
    pathway_matrices = []
    pathway_ids = []
    for pathway, compounds in pathways.items():
        single_pathway_matrix = mat.drop(mat.columns.difference(compounds), axis=1)
        if single_pathway_matrix.shape[1] >= 1:
            pathway_matrices.append(single_pathway_matrix.values)
            pathway_ids.append(pathway)

    scores = []
    for n, m in enumerate(pathway_matrices):
        kpca = KernelPCA(n_components=2, kernel="rbf")
        new_data = kpca.fit_transform(m)
        scores.append(new_data[:, 0])
    scores_df = pd.DataFrame(scores, columns=mat.index, index=pathways.keys())

    labels_binary = pd.factorize(mat["Group"])[0]
    pls2 = PLSRegression(n_components=5).fit(scores_df.T, labels_binary)
    coefs = list(zip([i[0] for i in pls2.coef_], scores_df.T.columns))
    coefs_df = pd.DataFrame(coefs, columns=["Coef", "ID"])
    coefs_df = coefs_df.sort_values(by="Coef", ascending=False, key=abs).reset_index()
    coefs_df["Coef"] = coefs_df["Coef"].abs()
    return coefs_df

def kpca_res(mat, pathways, testtype):
#     mat.loc[:, :-1] = StandardScaler().fit_transform(mat.iloc[:, :-1].to_numpy())
    pathway_matrices = []
    pathway_ids = []
    for pathway, compounds in pathways.items():
        single_pathway_matrix = mat.drop(mat.columns.difference(compounds), axis=1)
        if single_pathway_matrix.shape[1] >= 1:
            pathway_matrices.append(single_pathway_matrix.values)
            pathway_ids.append(pathway)

    scores = []
    for n, m in enumerate(pathway_matrices):
        kpca = KernelPCA(n_components=2, kernel="rbf")
        new_data = kpca.fit_transform(m)
        scores.append(new_data[:, 0])
    scores_df = pd.DataFrame(scores, columns=mat.index, index=pathways.keys())
    t_test_res = t_tests(scores_df.T, mat["Group"], "fdr_bh", testtype)
    return t_test_res


def fgsea_res(mat, pathways, dname):
    # Get rankings - SNR
    disease = mat[mat["Group"] == dname].drop(["Group"], axis=1)
    ctrl = mat[mat["Group"] != dname].drop(["Group"], axis=1)

    d_mean = np.mean(disease, axis=0)
    ctrl_mean = np.mean(ctrl, axis=0)
    d_std = np.std(disease, axis=0)
    ctrl_std = np.std(ctrl, axis=0)

    means = d_mean.subtract(ctrl_mean)
    stds = d_std + ctrl_std

    snr = means /stds
    snr_dict = snr.to_dict()
    
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_list = ro.ListVector(pathways)  # pathways
        r_ranks = ro.ListVector(snr_dict) # ranks 

    ro.r('''
    unl <- function(v){
    test <- unlist(v)
    return(test)}
    ''')

    r_unl = ro.globalenv['unl']
    r_ranks = r_unl(r_ranks)

    fgsea_r = importr('fgsea')
    fgsea_res = fgsea_r.fgsea(pathways=r_list, stats=r_ranks)
    
    df = pd.DataFrame(fgsea_res)
    df = df.T
    df = df.apply(pd.to_numeric, errors="ignore")
    df.columns = ["ID", "P-value", "P-adjust", "log2err", "ES", "NES", "coverage", "leadingEdge"]
    
    return df