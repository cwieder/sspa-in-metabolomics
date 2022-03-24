# sspa-in-metabolomics

This repository contains the code for the manuscript "Performance evaluation and applicability of single-sample pathway analysis methods to metabolomics data". 

## Data
Su et al. (2020) and Lloyd-Price et al (2019) metabolomics data can be found in the `datasets` folder. 

## Simulations and benchmarking

Code for the benchmarking analysis is in the `Benchmarking` folder. Simulations are expected to be run in parallel using a high-performance computing cluster. The code to run the simulations and calculate performance metrics is in the following files:
- Benchmarking/HPC_avg_norm_ranking_parallel.py
- Benchmarking/HPC_performance_metrics_parallel_effectsize.py
- Benchmarking/HPC_performance_metrics_parallel_noise.py

To run these files the following files are required in the same directory:
- Benchmarking/helper_functs.py: code to calculate performance metrics
- Benchmarking/methods.py: contains single-sample pathway analysis methods
- Benchmarking/process_met_data.py: processes metabolomics data prior to simulation
- Benchmarking/process_pathways.py: processes Reactome pathways into correct format
- Benchmarking/simulate_met_data.py: creates simulated metabolomics data

## IBD application
Code to create the IBD pathway-based correlation network is in the `Applications` folder in the `IBD_pathway_network.ipynb` notebook. This creates a .graphml file which can be imported into Cytoscape. 

Additional code for:
- Comparing the ARI achieved using metabolites vs pathways for clustering 
- PCA plots
- Random forest model based on kPCA pathway scores

can be found in the `sspa_IBD_applications.ipynb` notebook. 

## Cite
If you use this code please cite Wieder, Lai, and Ebbels (2022) [manuscript in preparation]. 