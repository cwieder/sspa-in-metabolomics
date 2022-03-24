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

    def process_stevens(self, id_type=None, normalisation=True):
        name_map = pd.read_csv("../datasets/name_map_stevens.csv", dtype="object")
        md_raw = pd.read_csv("../datasets/Stevens_metadata.txt", sep="\t")
        metadata_list = list(zip(md_raw['Factor Value[CurrentPMH]'], md_raw['Factor Value[Gender]'],
                                 md_raw['Factor Value[AgeAtBloodDraw]'],
                                 ['Over 75' if val not in ['<=55', '56-60', '61-65', '66-70', '71-75'] else 'Under 75'
                                  for val in md_raw['Factor Value[AgeAtBloodDraw]']]))
        metadata_dict = dict(zip(md_raw['Sample Name'].values, metadata_list))
        sample_status_dict = dict(zip(md_raw['Sample Name'].values, md_raw['Factor Value[CurrentPMH]']))

        replicate_samples = [k for k, v in metadata_dict.items() if v[0] not in ['Nonuser', 'E-only', 'E+P']]
        nonusers = [k for k, v in metadata_dict.items() if v[0] not in [np.nan, 'E-only', 'E+P']]
        # estrogen_only = [k for k, v in metadata_dict.items() if v[0] not in ['Nonuser', np.nan, 'E+P']]
        estrogen_progesterone = [k for k, v in metadata_dict.items() if v[0] not in ['Nonuser', 'E-only', np.nan]]
        # drop some control samples to make the dataset smaller and balance classes

        rng = np.random.default_rng()
        drop_nonusers = rng.choice(nonusers, 350, replace=False).tolist()
        # Get abundance matrix, transpose to n-samples by m-metabolites
        mat = pd.read_csv("../datasets/Stevens_matrix_named_compounds_only.csv", index_col=0, dtype=object)
        mat_nonusers_estrogen = mat.drop((replicate_samples + estrogen_progesterone + drop_nonusers), axis=1)
        self.raw_data = mat_nonusers_estrogen.iloc[:, 8:].T

        proc_mat = self.process_data(normalisation)
        proc_mat["Group"] = proc_mat.index.map(sample_status_dict)

        if id_type == "CHEBI":
            cpd_mapping_dict = dict(zip(name_map["Query"].tolist(), name_map["ChEBI"].tolist()))
            cpd_mapping_dict = {k: v for k, v in cpd_mapping_dict.items() if pd.notnull(v)}
            proc_mat_chebi = proc_mat.drop([i for i in proc_mat.columns if i not in cpd_mapping_dict.keys()], axis=1)
            proc_mat_chebi = proc_mat_chebi.rename(cpd_mapping_dict, axis=1)
            proc_mat_chebi["Group"] = proc_mat["Group"]
            self.data_proc = proc_mat_chebi

        elif id_type == "KEGG":
            cpd_mapping_dict = dict(zip(name_map["Query"].tolist(), name_map["KEGG"].tolist()))
            cpd_mapping_dict = {k: v for k, v in cpd_mapping_dict.items() if pd.notnull(v)}
            proc_mat_chebi = proc_mat.drop([i for i in proc_mat.columns if i not in cpd_mapping_dict.keys()], axis=1)
            proc_mat_chebi = proc_mat_chebi.rename(cpd_mapping_dict, axis=1)
            proc_mat_chebi["Group"] = proc_mat["Group"]
            self.data_proc = proc_mat_chebi
        else:
            self.data_proc = proc_mat


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

    def process_labbe(self, id_type=None):
        mat = pd.read_csv("../datasets/labbe_abundance.csv", index_col=0, header=1).T
        mapping = dict(zip(mat.columns.tolist(), mat.loc["KEGG", :].tolist()))
        mat = mat.rename(columns=mapping)
        mat = mat.loc[:, mat.columns.notnull()]
        mat = mat.loc[:, ~mat.columns.duplicated()]
        metadata = pd.read_csv("../datasets/labbe_metadata.txt", sep="\t")
        sample_name = [i[0:10] for i in metadata["Sample Name"]]
        diet = metadata["Factor Value[Genotype]"].tolist()
        metadata_dict = dict(zip(sample_name, diet))
        mat["Group"] = mat.index.map(metadata_dict)

        self.raw_data = mat
        groups = data["Group"]
        cancer_status = data["disease"]

        proc_mat = self.process_data()
        proc_mat["Group"] = proc_mat.index.map(groups)
        proc_mat["Cancer_status"] = proc_mat.index.map(cancer_status)
        mat_proc = utils.data_processing(mat, firstrow=6, firstcol=1)
        mat_proc["Group"] = mat_proc.index.map(metadata_dict)
        mat_proc = mat_proc.iloc[:, ~mat_proc.columns.duplicated()]
        self.proc_mat = mat_proc

    def process_yachida(self, id_type=None):
        name_map = pd.read_csv("../datasets/name_map_yachida.csv")
        data = pd.read_csv("../datasets/yachida_abundance.csv", index_col=0, header=0).T
        data = data.rename(columns={'Group': 'disease'})
        data = data.dropna(0)
        sample_disease_dict = dict(zip(data.index, data['disease']))
        data.columns = data.columns[0:4].tolist() + [col[0:6] for col in data.columns[4:]]

        removecols = []
        for i in data.columns.tolist():
            matchObj = re.search("^[C]\d{5}$", i)
            if not matchObj:
                removecols.append(i)

        data = data.drop(removecols[4:], axis=1)

        CRC_or_healthy_dict = dict.fromkeys(data.index.tolist())
        for k, v in sample_disease_dict.items():
            if v in ['Healthy']:
                CRC_or_healthy_dict[k] = "Healthy"
            elif v in ["Stage_I_II", "Stage_III_IV"]:
                CRC_or_healthy_dict[k] = "CRC"
            else:
                CRC_or_healthy_dict[k] = "Null"
        CRC_or_healthy = ["Healthy" if i in ["Healthy"] else "CRC" if i in ["Stage_I_II", "Stage_III_IV"] else "Null"
                          for i in data["disease"]]

        data.insert(1, "Group", CRC_or_healthy)

        data = data.iloc[:, ~data.columns.duplicated()]

        data = data[data.Group != "Null"]
        data = data.drop(["Gender", "BMI", "Age"], axis=1)
        self.raw_data = data.iloc[:, 2:]
        groups = data["Group"]
        cancer_status = data["disease"]

        proc_mat = self.process_data()
        proc_mat["Group"] = proc_mat.index.map(groups)
        proc_mat["Cancer_status"] = proc_mat.index.map(cancer_status)

        if id_type == "CHEBI":
            cpd_mapping_dict = dict(zip(name_map["Query"].tolist(), name_map["ChEBI"].tolist()))
            cpd_mapping_dict = {k: v for k, v in cpd_mapping_dict.items() if pd.notnull(v)}
            proc_mat_chebi = proc_mat.drop([i for i in proc_mat.columns if i not in cpd_mapping_dict.keys()], axis=1)
            proc_mat_chebi = proc_mat_chebi.rename(cpd_mapping_dict, axis=1)
            proc_mat_chebi["Group"] = proc_mat["Group"]
            proc_mat_chebi["Cancer_status"] = proc_mat["Cancer_status"]
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


def shapiro_test(proc_mat):
    # test for metabolite normal distribution
    data = proc_mat.iloc[:, :-2]
    pvals = []
    cv = []
    for col in range(0, data.shape[1]):
        s, p = shapiro(data.iloc[:, col])
        pvals.append(p)

    not_normal = [i for i in pvals if i < 0.05]
    print(len(not_normal))

    # data.boxplot(grid=False, showfliers=False)
    # plt.axis('off')
    # plt.show()


def pairplot(proc_mat, metadata):
    sns.set_style("ticks")
    goldman_pca = PCA(n_components=10).fit_transform(proc_mat)
    pca_df = pd.DataFrame(goldman_pca)
    pca_df.columns = ["PC " + str(i) for i in range(1, 11)]
    pca_df["Group"] = metadata.values
    sns.set_theme(style="ticks")
    sns.pairplot(pca_df.iloc[:, np.r_[0:5, 10]],
                 hue="Group",
                 diag_kind="auto")
    plt.show()


def pca_biplot(proc_mat, metadata):
    sns.set_style("ticks")
    goldman_pca = PCA(n_components=10).fit_transform(proc_mat)
    pca_df = pd.DataFrame(goldman_pca)
    pca_df.columns = ["PC " + str(i) for i in range(1, 11)]
    pca_df["Group"] = metadata.values
    sns.scatterplot(data=pca_df, x="PC 1", y="PC 2", hue="Group")
    plt.show()


def plt_scree():
    goldman_pca = PCA(n_components=15).fit(goldman_proc.iloc[:, :-2])
    print(goldman_pca.explained_variance_ratio_)
    cumulative_variance = [i / sum(goldman_pca.explained_variance_) for i in np.cumsum(goldman_pca.explained_variance_)]
    print(cumulative_variance)
    plt.plot([i + 1 for i in range(0, len(goldman_pca.explained_variance_))],
             [i for i in np.cumsum(goldman_pca.explained_variance_ratio_)], color="red")
    plt.bar([i + 1 for i in range(0, len(goldman_pca.explained_variance_))],
            [i for i in goldman_pca.explained_variance_ratio_])
    plt.title("Cumulative variance")
    plt.xlabel("PC")
    plt.ylabel("% variance explained")
    plt.suptitle("PCA with " + str(15) + " components")
    plt.tight_layout()
    plt.show()


if __name__ == "main":
    goldman_data = ProcData("goldman")
    goldman_data.process_goldman()
    goldman_proc = goldman_data.data_proc
    pca_biplot()

# goldman_data = ProcData("Stevens")
# goldman_data.process_labbe(id_type="CHEBI")
# goldman_proc = goldman_data.data_proc
# print(goldman_proc)
# print(goldman_proc.shape)
# goldman_proc.to_csv("/Users/cw2019/Documents/PhD/met-pathway-tutorial/Su_COVID_metabolomics_processed_CHEBI.csv")
#

# ibd_data = ProcData("IBD_HMP2")
# ibd_data.process_IBD(id_type="CHEBI")
# ibd_proc = ibd_data.data_proc
# print(ibd_proc)