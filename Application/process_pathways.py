import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class ProcessPathways:
    def __init__(self, name, infile, organism):
        self.name = name
        self.infile = infile
        self.organism = organism

    def process_reactome(self):
        # Process CHEBI to reactome data
        f = pd.read_csv("../pathway_databases/" + self.infile, sep="\t", header=None)
        f.columns = ['CHEBI', 'pathway_ID', 'link', 'pathway_name', 'evidence_code', 'species']
        f_filt = f[f.species == self.organism]
        name_dict = dict(zip(f_filt['pathway_ID'], f_filt['pathway_name']))

        groups = f_filt.groupby(['pathway_ID'])['CHEBI'].apply(list).to_dict()
        df = pd.DataFrame.from_dict(groups, orient='index', dtype="object")

        pathways_df = df.dropna(axis=0, how='all', subset=df.columns.tolist()[1:])
        pathways = pathways_df.index.tolist()
        pathway_dict = {}

        for pathway in pathways:
            pathway_compounds = list(set(pathways_df.loc[pathway, :].tolist()))
            pathway_compounds = [str(i) for i in pathway_compounds if str(i) != "None"]

            cpds = pathway_compounds[1:]
            if len(cpds) > 1:
                pathway_dict[pathway] = cpds
        return pathway_dict, name_dict

    
    def process_kegg(self):
        f = pd.read_csv("../pathway_databases/" + self.infile, index_col=0)
        name_dict = dict(zip(f.index, f['Pathway_name']))
        pathway_dict = {k: list(set(f.loc[k, '0':].tolist())) for k in list(name_dict.keys())}
        pathway_dict = {k: [i for i in v if pd.notnull(i)] for k, v in pathway_dict.items()}
        pathway_dict = {k: v for k, v in pathway_dict.items() if len(v) > 2}

        return pathway_dict, name_dict
        # remove dupes

    def process_reduced_pathway_set(self):
        # to process pathway sets created using the redundancy reduction algorithm
        f = pd.read_csv("../pathway_databases/" + self.infile, index_col=0, header=None, dtype='object')
        pathway_dict = {k: list(set(f.loc[k, '0':].tolist())) for k in f.index.tolist()}
        pathway_dict = {k: [i for i in v if pd.notnull(i)] for k, v in pathway_dict.items()}
        pathway_dict = {k: v for k, v in pathway_dict.items() if len(v) > 2}

        return pathway_dict


# if __name__ == "main":
#     R76 = ReactomePathways("R76", "../pathway_databases/ChEBI2Reactome_All_Levels.txt", "Homo sapiens")
#     pathway_dict = R76.process_reactome()
#
#     print(np.median([len(i) for i in pathway_dict.values()]))
#
#     plt.violinplot([len(i) for i in pathway_dict.values()])
#     plt.show()

# R76 = ProcessPathways("R76", "../pathway_databases/ChEBI2Reactome_All_Levels.txt", "Homo sapiens")
# pathway_names, pathway_dict = R76.process_reactome()
# df = pd.DataFrame.from_dict(pathway_names, orient='index')
# df["Pathway_name"] = df.index.map(pathway_dict)
# col = df.pop("Pathway_name")
# df.insert(0, "Pathway_name", col)
# # df.insert(0, 'Pathway_name', pathway_dict)
# df.to_csv("Reactome_human_pathways_compounds_R76.csv")