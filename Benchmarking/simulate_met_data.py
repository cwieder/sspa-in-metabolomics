import pandas as pd
import scipy.stats as stats
import numpy as np
from process_met_data import ProcData
from process_pathways import ProcessPathways
from sklearn.preprocessing import StandardScaler


class SimulateDataset:
    def __init__(self, input_data, metadata, pathways, upreg_paths, effect, noise=None):
        """
        Class to simulate data with n enriched pathways based on a real metabolomics dataset
        :param input_data: n samples * m metabolites processed abundance matrix
        :param upreg_paths:
        :param downreg_paths:
        """
        self.input_data = input_data
        self.metadata = metadata
        self.pathways = pathways
        self.upreg_paths = upreg_paths
        # self.downreg_paths = downreg_paths
        self.effect = int(effect)
        self.noise = noise
        self.upreg_paths_id = []
        # self.downreg_paths_id = []

    def generate_data(self):
        n_samples = self.input_data.shape[0]
        n_metabolites = self.input_data.shape[1]
        n_cases = self.input_data.shape[0] / 2
        n_controls = self.input_data.shape[0] / 2
        # wipe out original signals by adding values from a normal distribution
        # met_abundances = np.transpose(np.array([stats.norm.rvs(size=n_samples) for i in range(0, n_metabolites)]))
        # new_df = pd.DataFrame(met_abundances, columns=self.input_data.columns, index=self.input_data.index)

        # wipe out signals by permuting sample labels
        new_df = self.input_data
        rng = np.random.default_rng()
        metadata_new = rng.permutation(self.metadata.tolist())

        # add new signal
        pathway_idxs = list(np.random.randint(0, len(self.pathways), size=self.upreg_paths))
        selected_paths = [list(self.pathways.keys())[i] for i in pathway_idxs]
        upreg_paths = selected_paths[0:self.upreg_paths]
        self.upreg_paths_id = upreg_paths
        upreg_cpds = [self.pathways.get(key) for key in upreg_paths]
        upreg_cpds_filt = [item for sublist in upreg_cpds for item in sublist]
        upreg_cpds_filt = [i for i in upreg_cpds_filt if i in self.input_data.columns]
        upreg_indices = [self.input_data.columns.tolist().index(i) for i in upreg_cpds_filt]

        md_binary = np.array([1 if i == "COVID19 " else 0 for i in metadata_new])
        indices_0 = np.argwhere(md_binary == 0).ravel()  # control
        indices_1 = np.argwhere(md_binary == 1).ravel()  # disease

        # Increase or decrease signal in disease group
        if self.noise:
            n_alter_up = int(round(self.noise * len(upreg_indices), 0))
            # n_alter_down = int(round(self.noise * len(upreg_indices), 0))
            rng = np.random.default_rng()
            random_upreg_indices = rng.choice(upreg_indices, n_alter_up, replace=False)
            # random_downreg_indices = rng.choice(downreg_indices, n_alter_down, replace=False)
            new_df.iloc[indices_1, random_upreg_indices] = new_df.iloc[indices_1, random_upreg_indices] + self.effect
            # new_df.iloc[indices_1, downreg_indices] = new_df.iloc[indices_1, random_downreg_indices] - self.effect
        else:
            new_df.iloc[indices_1, upreg_indices] = new_df.iloc[indices_1, upreg_indices] + self.effect
            # new_df.iloc[indices_1, downreg_indices] = new_df.iloc[indices_1, downreg_indices] - self.effect
        # TODO if overlapping metabolites this up and down regulation could be a problem

        # standardscaler
        new_df.loc[:, :] = StandardScaler().fit_transform(new_df)

        # Add group labels
        new_df["Group"] = metadata_new

        return new_df


class SimulateDatasetNamed:
    def __init__(self, input_data, metadata, case_name, pathways, effect, scale=True,
                 upreg_paths=None, downreg_paths=None, noise=None, sample_pct=None):
        """
        Class to simulate data with n enriched pathways based on a real metabolomics dataset
        :param input_data: n samples * m metabolites processed abundance matrix
        :param upreg_paths:
        :param downreg_paths:
        """
        self.input_data = input_data
        self.metadata = metadata
        self.case_name = case_name
        self.pathways = pathways
        self.effect = float(effect)
        self.noise = noise
        self.sample_pct = sample_pct
        self.metadata_new = None
        self.scale = scale

        if downreg_paths is not None:
            self.downreg_paths = downreg_paths
            self.downreg_paths_id = downreg_paths
        else:
            self.downreg_paths = []
            self.downreg_paths_id = []
        if upreg_paths is not None:
            self.upreg_paths = upreg_paths
            self.upreg_paths_id = upreg_paths
        else:
            self.upreg_paths = []
            self.upreg_paths_id = []

    def generate_data(self):
        # wipe out signals by permuting sample labels
        new_df = self.input_data
        rng = np.random.default_rng()
        self.metadata_new = rng.permutation(self.metadata.tolist())
        upreg_cpds = [self.pathways.get(key) for key in self.upreg_paths]
        upreg_cpds_filt = [item for sublist in upreg_cpds for item in sublist]
        upreg_cpds_filt = [i for i in upreg_cpds_filt if i in self.input_data.columns]
        upreg_indices = [self.input_data.columns.tolist().index(i) for i in upreg_cpds_filt]
        downreg_cpds = [self.pathways.get(key) for key in self.downreg_paths]
        downreg_cpds_filt = [item for sublist in downreg_cpds for item in sublist]
        downreg_cpds_filt = [i for i in downreg_cpds_filt if i in self.input_data.columns]
        downreg_indices = [self.input_data.columns.tolist().index(i) for i in downreg_cpds_filt]

        md_binary = np.array([1 if i == self.case_name else 0 for i in self.metadata_new])
        indices_0 = np.argwhere(md_binary == 0).ravel()  # control
        indices_1 = np.argwhere(md_binary == 1).ravel()  # disease
        # print(indices_1)
        # Increase or decrease signal in disease group
        # Log2 FC = 4
        if self.noise:
            n_alter_up = int(round(self.noise * len(upreg_indices), 0))
            # n_alter_down = int(round(self.noise * len(upreg_indices), 0))
            rng = np.random.default_rng()
            random_upreg_indices = rng.choice(upreg_indices, n_alter_up, replace=False)
            # random_downreg_indices = rng.choice(downreg_indices, n_alter_down, replace=False)
            new_df.iloc[indices_1, random_upreg_indices] = new_df.iloc[indices_1, random_upreg_indices] + self.effect
            # new_df.iloc[indices_1, downreg_indices] = new_df.iloc[indices_1, random_downreg_indices] - self.effect
            # standardscaler
            if self.scale:
#                 new_df.iloc[:, random_upreg_indices] = StandardScaler().fit_transform(new_df.iloc[:, random_upreg_indices].to_numpy())
                new_df.iloc[:, :] = StandardScaler().fit_transform(new_df.to_numpy())
            else:
                pass
        else:
            new_df.iloc[indices_1, upreg_indices] = new_df.iloc[indices_1, upreg_indices] + self.effect
            # new_df.iloc[indices_1, downreg_indices] = new_df.iloc[indices_1, downreg_indices] - self.effect
            # standardscaler
            if self.scale:
#                 new_df.iloc[:, upreg_indices] = StandardScaler().fit_transform(new_df.iloc[:, upreg_indices].to_numpy())
                new_df.iloc[:, :] = StandardScaler().fit_transform(new_df.to_numpy())
            else:
                pass

        # removes x% of samples (class balanced)
        if self.sample_pct is not None:
            md_dict = dict(zip(new_df.index, self.metadata_new))
            case_dict = {k: v for k, v in md_dict.items() if v == self.case_name}
            control_dict = {k: v for k, v in md_dict.items() if v != self.case_name}
            n_drop = int((new_df.shape[0] * self.sample_pct) / 2)  # drop this number of each class
            rng = np.random.default_rng()
            drop_cases = rng.choice(list(case_dict.keys()), n_drop, replace=False).tolist()
            drop_controls = rng.choice(list(control_dict.keys()), n_drop, replace=False).tolist()
            new_df = new_df.drop(labels=drop_cases + drop_controls)
            self.metadata_new = [v for k, v in md_dict.items() if k not in drop_cases + drop_controls]

        # Add group labels
        new_df["Group"] = self.metadata_new

        return new_df


class SimulateDatasetConfouderUncorrelated(SimulateDatasetNamed):
    # inherits from the parent class simulate dataset named
    def __init__(self, input_data, metadata, case_name, pathways, effect, upreg_paths=None, noise=None,
                 confounder_paths=None):
        super().__init__(input_data, metadata, case_name, pathways, effect, upreg_paths=None, noise=None)
        if upreg_paths is not None:
            self.upreg_paths = upreg_paths
            self.upreg_paths_id = upreg_paths
        else:
            self.upreg_paths = []
            self.upreg_paths_id = []
        self.confounders = confounder_paths

    def generate_data(self):
        sim_data = super().generate_data()
        md = sim_data["Group"]
        sim_data = sim_data.iloc[:, :-1]

        # select half the samples for confounder signal
        rng = np.random.default_rng()
        random_samples = rng.choice(self.input_data.index.tolist(), int(self.input_data.shape[0] / 2), replace=False)
        samples_indices = np.where(sim_data.index.isin(random_samples))[0]

        upreg_cpds = [self.pathways.get(key) for key in self.confounders]
        upreg_cpds_filt = [item for sublist in upreg_cpds for item in sublist]
        upreg_cpds_filt = [i for i in upreg_cpds_filt if i in self.input_data.columns]
        upreg_indices = [self.input_data.columns.tolist().index(i) for i in upreg_cpds_filt]

        sim_data.iloc[samples_indices, upreg_indices] = sim_data.iloc[samples_indices, upreg_indices] + self.effect

        # standardscaler
        sim_data.loc[:, :] = StandardScaler().fit_transform(sim_data)

        # Add group labels
        sim_data["Group"] = md
        sim_data["Confound_status"] = [1 if i in random_samples else 0 for i in md.index]
        return sim_data


class SimulateDatasetConfouderCorrelated(SimulateDatasetNamed):

    def __init__(self, input_data, metadata, case_name, pathways, effect, upreg_paths=None, noise=None,
                 confounder_paths=None):
        """
        Class to simulate data with confounder signal which is correlated to the phenotype
        :param input_data:
        :param metadata:
        :param case_name:
        :param pathways:
        :param effect:
        :param upreg_paths:
        :param noise:
        :param confounder_paths:
        """
        super().__init__(input_data, metadata, case_name, pathways, effect, upreg_paths=None, noise=None)

        if upreg_paths is not None:
            self.upreg_paths = upreg_paths
            self.upreg_paths_id = upreg_paths
        else:
            self.upreg_paths = []
            self.upreg_paths_id = []
        self.confounders = confounder_paths

    def generate_data(self):
        sim_data = super().generate_data()
        ctrl_samples = sim_data[sim_data["Group"] != self.case_name]
        cond_samples = sim_data[sim_data["Group"] == self.case_name]
        md = sim_data["Group"]
        sim_data = sim_data.iloc[:, :-1]

        # 75% of disease samples and 25% of control samples for the confounder signal
        rng = np.random.default_rng()
        random_samples_ctrl = rng.choice(ctrl_samples.index.tolist(), int(ctrl_samples.shape[0] * 0.25),
                                         replace=False).tolist()
        random_samples_cond = rng.choice(cond_samples.index.tolist(), int(cond_samples.shape[0] * 0.75),
                                         replace=False).tolist()
        samples_indices = np.where(sim_data.index.isin(random_samples_cond + random_samples_ctrl))[0]

        upreg_cpds = [self.pathways.get(key) for key in self.confounders]
        upreg_cpds_filt = [item for sublist in upreg_cpds for item in sublist]
        upreg_cpds_filt = [i for i in upreg_cpds_filt if i in self.input_data.columns]
        upreg_indices = [self.input_data.columns.tolist().index(i) for i in upreg_cpds_filt]

        sim_data.iloc[samples_indices, upreg_indices] = sim_data.iloc[samples_indices, upreg_indices] + (self.effect*2)

        # standardscaler
        sim_data.loc[:, :] = StandardScaler().fit_transform(sim_data)

        # Add group labels
        sim_data["Group"] = md
        # sim_data["Confound_status"] = [1 if i in random_samples_cond+random_samples_ctrl else 0 for i in md.index]
        return sim_data

