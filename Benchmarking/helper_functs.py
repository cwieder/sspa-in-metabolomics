import pandas as pd
import methods
import scipy.stats as stats
from sklearn.metrics import roc_auc_score

def get_rankings_univariate(true_pathway_ids, mat, pathways, dname):
    svd_ttest_res = methods.svd_score_ttest(mat, pathways)
    svd_ttest_res["rank"] = stats.rankdata([i for i in svd_ttest_res["P-value"]], method="min")
    svd_res_rankings = [svd_ttest_res[svd_ttest_res['Metabolite'] == i]["rank"].tolist()[0] for i in true_pathway_ids]

    gsva_ttest_res = methods.gsva_ttest(mat, pathways)
    gsva_ttest_res["rank"] = stats.rankdata([i for i in gsva_ttest_res["P-value"]], method="min")
    gsva_res_rankings = [gsva_ttest_res[gsva_ttest_res['Metabolite'] == i]["rank"].tolist()[0] for i in true_pathway_ids]

    ssgsea_ttest_res = methods.ssgsea_results(mat, pathways)
    ssgsea_ttest_res["rank"] = stats.rankdata([i for i in ssgsea_ttest_res["P-value"]], method="min")
    ssgsea_res_rankings = [ssgsea_ttest_res[ssgsea_ttest_res['Metabolite'] == i]["rank"].tolist()[0] for i in true_pathway_ids]

    zscore_res = methods.zscore_res(mat, pathways)
    zscore_res["rank"] = stats.rankdata([i for i in zscore_res["P-value"]], method="min")
    zscore_res_rankings = [zscore_res[zscore_res['Metabolite'] == i]["rank"].tolist()[0] for i in true_pathway_ids]

    singscore_res = methods.singscore_ttest(mat, pathways)
    singscore_res["rank"] = stats.rankdata([i for i in singscore_res["P-value"]], method="min")
    singscore_res_rankings = [singscore_res[singscore_res['Metabolite'] == i]["rank"].tolist()[0] for i in true_pathway_ids]

    cluster_res = methods.cluster_ttest(mat, pathways)
    cluster_res["rank"] = stats.rankdata([i for i in cluster_res["P-value"]], method="min")
    cluster_res_rankings = [cluster_res[cluster_res['Metabolite'] == i]["rank"].tolist()[0] for i in true_pathway_ids]
    
    clusterproj_res = methods.clusterproj_ttest(mat, pathways)
    clusterproj_res["rank"] = stats.rankdata([i for i in clusterproj_res["P-value"]], method="min")
    clusterproj_res_rankings = [clusterproj_res[clusterproj_res['Metabolite'] == i]["rank"].tolist()[0] for i in true_pathway_ids]

    kpca_res = methods.kpca_res(mat, pathways)
    kpca_res["rank"] = stats.rankdata([i for i in kpca_res["P-value"]], method="min")
    kpca_res_rankings = [kpca_res[kpca_res['Metabolite'] == i]["rank"].tolist()[0] for i in true_pathway_ids]

    ora_res = methods.ora_results(mat, pathways)
    ora_res = ora_res.rename(columns={"ID": "Metabolite"})
    ora_res["rank"] = stats.rankdata([i for i in ora_res["P-value"]], method="min")

    ora_res_rankings = []
    for i in true_pathway_ids:
        try:
            r = ora_res[ora_res['Metabolite'] == i]["rank"].tolist()[0]
            ora_res_rankings.append(r)
        except IndexError:
            ora_res_rankings.append("NA")

    gsea_res = methods.fgsea_res(mat, pathways, dname)
    gsea_res = gsea_res.rename(columns={"ID": "Metabolite", "fdr": "P-adjust", "pval": "P-value"})
    gsea_res["rank"] = stats.rankdata([i for i in gsea_res["P-value"]], method="min")
    gsea_res_rankings = [gsea_res[gsea_res['Metabolite'] == i]["rank"].tolist()[0] for i in true_pathway_ids]

    res_df = pd.DataFrame(index=true_pathway_ids)
    # res_df["Direction"] = res_df.index.map(sims_dict)
    res_df["SVD"] = [int(i) / len(svd_ttest_res) for i in svd_res_rankings]
    res_df["ssgsea"] = [int(i) / len(ssgsea_ttest_res) for i in ssgsea_res_rankings]
    res_df["GSVA"] = [int(i) / len(gsva_ttest_res) for i in gsva_res_rankings]
    res_df["z-score"] = [int(i) / len(zscore_res) for i in zscore_res_rankings]
    res_df["singscore"] = [int(i) / len(singscore_res) for i in singscore_res_rankings]
    # res_df["MBPLS coef"] = [int(i) / len(mbpls_res) for i in mbpls_res_coef_rankings]
    res_df["cluster"] = [int(i) / len(cluster_res) for i in cluster_res_rankings]
    res_df["clusterproj"] = [int(i) / len(clusterproj_res) for i in clusterproj_res_rankings]
    res_df["KPCA"] = [int(i) / len(kpca_res) for i in kpca_res_rankings]
    res_df["GSEA"] = [int(i) / len(gsea_res) for i in gsea_res_rankings]
    res_df["ORA"] = [int(i) / len(ora_res) if i != "NA" else pd.NA for i in ora_res_rankings]

    rankings_sum = res_df.mean(axis=0).tolist()
    return res_df, rankings_sum



def performance_metrics(ttest_res, thresh, enrich_paths, coverage_dict, overlap_thresh=1):
    all_enriched_metabs = [coverage_dict[i] for i in enrich_paths]
    all_enriched_metabs = [item for sublist in all_enriched_metabs for item in sublist]
    
    try:
        ttest_res["Overlap"] = [methods.overlap_coefficient(coverage_dict[i], all_enriched_metabs) for i in ttest_res["Metabolite"]]
    except ZeroDivisionError:
        ttest_res["Overlap"] = [0 for i in ttest_res["Metabolite"]]
        
    sig_paths = ttest_res[ttest_res["P-adjust"] <= thresh]["Metabolite"].tolist()
    non_sig_paths = ttest_res[ttest_res["P-adjust"] > thresh]["Metabolite"].tolist()
    enriched_all = ttest_res[ttest_res["Overlap"] >= overlap_thresh]["Metabolite"].tolist()  # enriched pathways including those that overlap
    #enriched_all = enrich_paths

    TP = len(list(set(set(sig_paths) & set(enriched_all))))
    FP = len([i for i in sig_paths if i not in enriched_all])
    FN = len([i for i in enriched_all if i not in sig_paths])
    TN = len([i for i in non_sig_paths if i not in enriched_all])
    
    try:
        recall = TP / (TP + FN)
    except ZeroDivisionError:
        recall = pd.NA

    try:
        precision = TP / (TP + FP)
    except ZeroDivisionError:
        precision = pd.NA
    
#     try:
#         accuracy = (TP + TN) / (TP + TN + FP + FN)
#     except ZeroDivisionError:
#         accuracy = pd.NA
    
    bin_true = [1 if i in enriched_all else 0 for i in ttest_res["Metabolite"]]
    bin_pred = [1 if i <= thresh else 0 for i in ttest_res["P-adjust"]]

    try:
        auc = roc_auc_score(bin_true, bin_pred)
    except ValueError:
        auc = pd.NA
    res_df = pd.DataFrame({"Recall": [recall], "Precision": [precision], "AUC": [auc]})
    return res_df

def get_metrics(true_pathway_ids, mat, pathways, pval_thresh, coverage_dict, overlap_t, dname, testtype="ttest"):
    svd_ttest_res = methods.svd_score_ttest(mat, pathways, testtype)
    svd_metrics = performance_metrics(svd_ttest_res, pval_thresh, true_pathway_ids, coverage_dict, overlap_thresh=overlap_t)

    gsva_ttest_res = methods.gsva_ttest(mat, pathways, testtype)
    gsva_metrics = performance_metrics(gsva_ttest_res, pval_thresh, true_pathway_ids, coverage_dict, overlap_thresh=overlap_t)

    ssgsea_ttest_res = methods.ssgsea_results(mat, pathways, testtype)
    ssgsea_metrics = performance_metrics(ssgsea_ttest_res, pval_thresh, true_pathway_ids, coverage_dict, overlap_thresh=overlap_t)

    zscore_res = methods.zscore_res(mat, pathways, testtype)
    zscore_metrics = performance_metrics(zscore_res, pval_thresh, true_pathway_ids, coverage_dict, overlap_thresh=overlap_t)

    # singscore_res = methods.singscore_ttest(mat, pathways)
    # singscore_metrics = performance_metrics(singscore_res, pval_thresh, true_pathway_ids, coverage_dict,
    #                                      overlap_thresh=overlap_t)

    cluster_res = methods.cluster_ttest(mat, pathways, testtype)
    cluster_metrics = performance_metrics(cluster_res, pval_thresh, true_pathway_ids, coverage_dict,
                                            overlap_thresh=overlap_t)
    
    clusterproj_res = methods.clusterproj_ttest(mat, pathways, testtype)
    clusterproj_metrics = performance_metrics(clusterproj_res, pval_thresh, true_pathway_ids, coverage_dict,
                                            overlap_thresh=overlap_t)

    kpca_res = methods.kpca_res(mat, pathways, testtype)
    kpca_metrics = performance_metrics(kpca_res, pval_thresh, true_pathway_ids, coverage_dict,
                                          overlap_thresh=overlap_t)
#     ora_res = methods.ora_results(mat, "../pathway_databases/Reactome_human_pathways_compounds_R76_nonredundant.csv")
    ora_res = methods.ora_results(mat, pathways)
    # CHANGE ME
    ora_allpaths = pd.DataFrame(columns=ora_res.columns)
    ora_allpaths["ID"] = [i for i in pathways.keys() if i not in ora_res["ID"].tolist()]
    ora_allpaths["P-value"] = 1
    ora_allpaths["P-adjust"] = 1
    ora_res = pd.concat([ora_res, ora_allpaths], ignore_index=True)
    ora_res = ora_res.rename(columns={"ID": "Metabolite"})
    ora_metrics = performance_metrics(ora_res, pval_thresh, true_pathway_ids, coverage_dict, overlap_thresh=overlap_t)

#     gsea_gmt = "../pathway_databases/Reactome_human_pathways_compounds_R76_nonredundant.gmt"
    gsea_res = methods.fgsea_res(mat, pathways, dname)
    gsea_res = gsea_res.rename(columns={"ID": "Metabolite"})
    gsea_metrics = performance_metrics(gsea_res, pval_thresh, true_pathway_ids, coverage_dict, overlap_thresh=overlap_t)

    final_df = pd.concat([svd_metrics, ssgsea_metrics, zscore_metrics, gsva_metrics, cluster_metrics, clusterproj_metrics, kpca_metrics, gsea_metrics, ora_metrics],
                         keys=["SVD", "ssGSEA", "zscore", "GSVA", "cluster", "clusterproj", "kPCA", "GSEA", "ORA"])
    return final_df


