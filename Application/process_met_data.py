import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fancyimpute import IterativeSVD
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import shapiro, variation
import re


class ProcData:
    def __init__(self, name):
        self.name = name
        self.raw_data = []
        self.data_proc = []

    def process_data(self, normalisation=True):
        """
        General untargeted metabolomics data processing function
        :return: processed abundance matrix
        """

        raw_mat = self.raw_data
        raw_mat = raw_mat.replace(',', '', regex=True)
        raw_mat = raw_mat.apply(pd.to_numeric)
        # sns.heatmap(raw_mat.isnull(), cbar=False)
        # plt.show()
        # remove 50% nan compounds
        proc_mat = raw_mat.loc[:, raw_mat.isin([' ', np.nan, 0]).mean() < 0.7]

        if normalisation:
            # normalise samples with median fold change (PQN)
            met_median = proc_mat.median(axis=0, skipna=True)  # median value for each metabolite
            scale_mat = proc_mat.divide(met_median, axis=1)  # scale the matrix by the metabolite median
            samp_median = scale_mat.median(axis=1, skipna=True)  # median value for each sample
            proc_mat = proc_mat.divide(samp_median, axis=0)  # scale by sample median
        else:
            pass
        # impute SVD
        proc_mat.loc[:, :] = StandardScaler().fit_transform(proc_mat.to_numpy())
        proc_mat.loc[:, :] = IterativeSVD(verbose=False).fit_transform(proc_mat.to_numpy())

        # add min value + 1 as constant and log transform
        min_val = np.amin(proc_mat.to_numpy())
        proc_mat.loc[:, :] = proc_mat.to_numpy() + (abs(min_val) + 1)
        proc_mat.loc[:, :] = np.log2(proc_mat)
        # can remove mets with CV > 0.3

        # center and scale
        # TODO remove outlier metabolites?
        proc_mat.loc[:, :] = StandardScaler().fit_transform(proc_mat.to_numpy(dtype=float))

        return proc_mat




    def process_goldman(self, id_type=None, normalisation=True):
        """
        Specific function to process Goldman covid data
        :param id_type: metabolite identifier name
        :param chebi_only: Keep only compounds with CHEBI ID
        :return: processed abundance matrix
        """
        covid = pd.read_csv("../datasets/Goldman_metabolites.csv", index_col=0)
        covid = covid.drop([i for i in covid.index if i.endswith("AC")], axis=0)
        covid_metadata = pd.read_csv("../datasets/Goldman_clinical_data.csv")
        covid_namemap = pd.read_csv("../datasets/name_map_goldman.csv", dtype="object")
        blood_draw1 = covid_metadata[covid_metadata["Blood draw time point"] == "T1"]
        who_status = dict(zip(blood_draw1["Study Subject ID"].tolist(), blood_draw1["Who Ordinal Scale"].tolist()))
        who_map = {}
        for i in covid.index:
            if i.startswith("INCOV"):
                if who_status[i[:-3]] in ["3", "4"]:
                    who_map[i] = "3-4"
                elif who_status[i[:-3]] in ["5", "6", "7"]:
                    who_map[i] = "5-7"
                else:
                    who_map[i] = "1-2"
            else:
                who_map[i] = 0

        covid["WHO_status"] = covid.index.map(who_map)
        covid = covid[['WHO_status'] + [col for col in covid.columns if col != 'WHO_status']]

        # remove outliers
        covid.drop(["INCOV064-BL", "INCOV028-BL", "INCOV090-BL"], axis=0, inplace=True)
        self.raw_data = covid.iloc[:, 2:]
        groups = covid.iloc[:, 1]
        who_status = covid.iloc[:, 0]
        proc_mat = self.process_data(normalisation)
        proc_mat["Group"] = proc_mat.index.map(groups)
        proc_mat["WHO_status"] = proc_mat.index.map(who_status)

        if id_type == "CHEBI":
            cpd_mapping_dict = dict(zip(covid_namemap["Query"].tolist(), covid_namemap["ChEBI"].tolist()))
            cpd_mapping_dict = {k: v for k, v in cpd_mapping_dict.items() if pd.notnull(v)}
            proc_mat_chebi = proc_mat.drop([i for i in proc_mat.columns if i not in cpd_mapping_dict.keys()], axis=1)
            proc_mat_chebi = proc_mat_chebi.rename(cpd_mapping_dict, axis=1)
            proc_mat_chebi["WHO_status"] = proc_mat["WHO_status"]
            proc_mat_chebi["Group"] = proc_mat["Group"]
            self.data_proc = proc_mat_chebi
            
        elif id_type == "KEGG":
            cpd_mapping_dict = dict(zip(covid_namemap["Query"].tolist(), covid_namemap["KEGG"].tolist()))
            cpd_mapping_dict = {k: v for k, v in cpd_mapping_dict.items() if pd.notnull(v)}
            proc_mat_chebi = proc_mat.drop([i for i in proc_mat.columns if i not in cpd_mapping_dict.keys()], axis=1)
            proc_mat_chebi = proc_mat_chebi.rename(cpd_mapping_dict, axis=1)
            proc_mat_chebi["WHO_status"] = proc_mat["WHO_status"]
            proc_mat_chebi["Group"] = proc_mat["Group"]
            self.data_proc = proc_mat_chebi
        else:
            self.data_proc = proc_mat


    def process_IBD(self, id_type=None, normalisation=True):
        name_map = pd.read_csv("../datasets/IBD_name_map.csv")
        md = pd.read_csv("../datasets/hmp2_metadata.csv", dtype=object)
        metabolomics = pd.read_csv("../datasets/HMP2_metabolomics.csv.zip", dtype=object)
        # Keep only features with a valid HMDB ID
        metab_hmdb = metabolomics.dropna(subset=["HMDB (*Representative ID)"])
        metab_hmdb.index = metab_hmdb["HMDB (*Representative ID)"]
        metabolomics_samples = metab_hmdb.iloc[:, 7:]
        # get metadata for samples
        md_metabolomics = md[md["External ID"].isin(metabolomics_samples.columns)]
        
#         visit1 = md_metabolomics[(md_metabolomics["week_num"] == "0.0") & (
#                         md_metabolomics["data_type"] == "metabolomics")]["External ID"].tolist()
        
        visit1 = md_metabolomics[md_metabolomics["data_type"] == "metabolomics"]["External ID"].tolist()
        metabolomics_samples = metabolomics_samples.loc[:, visit1].T
        
#         ibd_status = dict(zip(md_metabolomics[(md_metabolomics["week_num"] == "0.0") & (
#                         md_metabolomics["data_type"] == "metabolomics")]["External ID"].tolist(), md_metabolomics[(md_metabolomics["week_num"] == "0.0") & (
#                         md_metabolomics["data_type"] == "metabolomics")]["diagnosis"].tolist()))
        
        ibd_status = dict(zip(md_metabolomics[md_metabolomics["data_type"] == "metabolomics"]["External ID"].tolist(), md_metabolomics[md_metabolomics["data_type"] == "metabolomics"]["diagnosis"].tolist()))
        
        md_dict = {k: "IBD" if v in ["UC", "CD"] else "non-IBD" for k, v in ibd_status.items()}
        self.raw_data = metabolomics_samples
        proc_mat = self.process_data(normalisation)
        proc_mat["Group"] = proc_mat.index.map(md_dict)
        proc_mat["IBD_status"] = proc_mat.index.map(ibd_status)

        if id_type == "CHEBI":
            cpd_mapping_dict = dict(zip(name_map["Query"].tolist(), name_map["ChEBI"].tolist()))
            cpd_mapping_dict = {k: v for k, v in cpd_mapping_dict.items() if pd.notnull(v)}
            proc_mat_chebi = proc_mat.drop([i for i in proc_mat.columns if i not in cpd_mapping_dict.keys()], axis=1)
            proc_mat_chebi = proc_mat_chebi.rename(cpd_mapping_dict, axis=1)
            proc_mat_chebi["Group"] = proc_mat["Group"]
            proc_mat_chebi["IBD_status"] = proc_mat["IBD_status"]
            self.data_proc = proc_mat_chebi
        elif id_type == "KEGG":
            cpd_mapping_dict = dict(zip(name_map["Query"].tolist(), name_map["KEGG"].tolist()))
            cpd_mapping_dict = {k: v for k, v in cpd_mapping_dict.items() if pd.notnull(v)}
            proc_mat_chebi = proc_mat.drop([i for i in proc_mat.columns if i not in cpd_mapping_dict.keys()], axis=1)
            proc_mat_chebi = proc_mat_chebi.rename(cpd_mapping_dict, axis=1)
            proc_mat_chebi["Group"] = proc_mat["Group"]
            proc_mat_chebi["IBD_status"] = proc_mat["IBD_status"]
            self.data_proc = proc_mat_chebi
        else:
            self.data_proc = proc_mat



