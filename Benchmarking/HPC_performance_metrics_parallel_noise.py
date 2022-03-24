import numpy as np
import pandas as pd
from process_met_data import ProcData
from process_pathways import ProcessPathways
from simulate_met_data import SimulateDataset, SimulateDatasetNamed, SimulateDatasetConfouderCorrelated
import helper_functs
import sys


goldman_data = ProcData("Goldman")
goldman_data.process_goldman(id_type="CHEBI")
goldman_proc_orig = goldman_data.data_proc

# Load Reactome pathway dictionary
R76 = ProcessPathways("R76", "ChEBI2Reactome_All_Levels.txt", "Homo sapiens")
pathway_dict, pathway_names = R76.process_reactome()

# Remove large and uninformative pathways
remove_paths = ["R-HSA-1430728", "R-HSA-1643685", "R-HSA-382551"]
pathway_dict = {k: v for k, v in pathway_dict.items() if k not in remove_paths}

# Remove pathways not present in the dataset
compounds_present = goldman_proc_orig.columns.tolist()
pathways_present = {k: v for k, v in pathway_dict.items() if len([i for i in compounds_present if i in v]) > 1}
path_coverage = {k: [i for i in v if i in compounds_present] for k, v in pathways_present.items()}

pathways_greater_than_5 = {}
for k, v in pathways_present.items():
    if len(set(v).intersection(compounds_present)) >= 3:
        pathways_greater_than_5[k] = v

nr_set = {k: v for k, v in pathways_present.items() if k in ['R-HSA-109582', 'R-HSA-1237112', 'R-HSA-1266738', 'R-HSA-1483115', 'R-HSA-156582', 'R-HSA-156590', 'R-HSA-1614603', 'R-HSA-1660662', 'R-HSA-189200', 'R-HSA-196071', 'R-HSA-211981', 'R-HSA-3296197', 'R-HSA-352230', 'R-HSA-5619084', 'R-HSA-5683826', 'R-HSA-73614', 'R-HSA-77108', 'R-HSA-917937']}


res_dfs = []

# change effect size - log fold change
noise_level = [0, 0.2, 0.4, 0.6, 0.8, 1]
# multiple enriched pathways
rankings_res = []

for e in noise_level:
    print(e)
    if e == 0:
        # simulate data
        rng = np.random.default_rng()
        random_pathways = rng.choice(list(pathways_greater_than_5.keys()), 3, replace=False)
        goldman_sim_iter = SimulateDatasetNamed(goldman_data.data_proc.iloc[:, :-2],
                                                goldman_data.data_proc["Group"],
                                                "COVID19 ",
                                                pathways_present, 0, upreg_paths=random_pathways)
        iter_sim = goldman_sim_iter.generate_data()
        upreg_paths = goldman_sim_iter.upreg_paths_id
        downreg_paths = goldman_sim_iter.downreg_paths_id
        sims_dict = {}
        for i in [*upreg_paths, *downreg_paths]:
            if i in upreg_paths:
                sims_dict[i] = "upreg"
            else:
                sims_dict[i] = "downreg"

        # run methods
        res = helper_functs.get_metrics(sims_dict.keys(), iter_sim, pathways_present, 0.05, path_coverage, 0.5, "COVID19 ")
        res_dfs.append(res)
    else:
        # simulate data
        rng = np.random.default_rng()
        random_pathways = rng.choice(list(pathways_greater_than_5.keys()), 3, replace=False)
        goldman_sim_iter = SimulateDatasetNamed(goldman_data.data_proc.iloc[:, :-2],
                                                goldman_data.data_proc["Group"],
                                                "COVID19 ",
                                                pathways_present, 1, upreg_paths=random_pathways, noise=e)
        iter_sim = goldman_sim_iter.generate_data()
        upreg_paths = goldman_sim_iter.upreg_paths_id
        downreg_paths = goldman_sim_iter.downreg_paths_id
        sims_dict = {}
        for i in [*upreg_paths, *downreg_paths]:
            if i in upreg_paths:
                sims_dict[i] = "upreg"
            else:
                sims_dict[i] = "downreg"

        # run methods
        res = helper_functs.get_metrics(sims_dict.keys(), iter_sim, pathways_present, 0.05, path_coverage, 0.5, "COVID19 ")
        res_dfs.append(res)


final_df = pd.concat(res_dfs, keys=noise_level)

final_df.to_csv("tempmetrics/metrics_3upreg_noise_200_50pct_effect1_" + str(sys.argv[1]) + ".csv")
