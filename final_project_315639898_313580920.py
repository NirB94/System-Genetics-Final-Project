import numpy as np
import pandas as pd
from statsmodels.stats.multitest import fdrcorrection
import preprocessing as pp
import HW3
import regression
import CausalityAnalysis as ca

GENOTYPES = 'genotypes.xls'
PHENOTYPE_PATH = 'phenotypes.xls'
MGI = 'MGI_Coordinates.Build37.rpt.txt'
PHENOTYPES = {"Morphine response (50 mg/kg ip), locomotion (open field) from 45-60 min after injection in an activity"
              " chamber for males [cm]": 970,
              "Morphine response (50 mg/kg ip), locomotion (open field) from 45-60 min after injection in an activity"
              " chamber for females [cm]": 1224,
              "Morphine response (50 mg/kg ip), locomotion (open field) from 45-60 min after injection in an activity"
              " chamber for males and females [cm]": 1478}
REVERSE_PHENOTYPES = {value: key for key, value in PHENOTYPES.items()}


def calculate_qtls(genotypes, phenotypes_path):
    p_vals_per_phenotype = dict()
    for name, ID in PHENOTYPES.items():
        phenotype = regression.create_phenotype_df(phenotypes_path, ID)
        filtered_genotypes = HW3.filtering(genotypes, extract_strains(phenotype))
        p_vals = regression.regression_model(phenotype, filtered_genotypes)
        p_vals['phenotype'] = ID
        p_vals_per_phenotype[name] = p_vals

    correction_df = pd.concat([p_vals_per_phenotype[key] for key in p_vals_per_phenotype.keys()])
    correction_df.fillna(1, inplace=True)
    correction_df['P_value'] = fdrcorrection(correction_df['P_value'], 0.05)[1]

    for name, ID in PHENOTYPES.items():
        corrected_phenotype = correction_df[correction_df['phenotype'] == ID]
        corrected_phenotype['-log(P_value)'] = -np.log10(corrected_phenotype['P_value'])
        p_vals_per_phenotype[name] = corrected_phenotype

    return p_vals_per_phenotype


def extract_strains(phenotype):
    strains = []
    for name in phenotype.index:
        if 'BXD' in name:
            strains.append(name)
    return strains


def all_manhattan_plots(phenotypes_p_vals_df):
    for name in phenotypes_p_vals_df.keys():
        temp_df = phenotypes_p_vals_df[name]
        regression.create_manhattan_plot(temp_df, name)


def remove_insignificant(regression_dict):
    for phenotype in regression_dict.keys():
        temp_df = regression_dict[phenotype]
        regression_dict[phenotype] = temp_df[temp_df['P_value'] <= 0.05]
    return regression_dict


def get_key(value, dic):
    for key in dic.keys():
        if dic[key] == value:
            return key
    return None


if __name__ == '__main__':
    liver_accession = 'GSE17522'
    hypothalamus_accession = 'GSE36674'
    genotypes = pd.read_excel(GENOTYPES, skiprows=1).drop(0, axis=1)
    mgi = pd.read_csv(MGI, sep='\t', header=0)

    ### Section 2 ###
    print("Beginning Section 2")
    # hypo_processed_data = pp.run_preprocessing(hypothalamus_accession, liver=False)  # Provided as pre-prepared files.
    # liver_processed_data = pp.run_preprocessing(liver_accession)
    hypo_processed_data = pd.read_csv('Hypothalamus_data_after_processing.csv', index_col=[0])
    liver_processed_data = pd.read_csv('Liver_data_after_processing.csv', index_col=[0])

    ### Section 3 ###
    print("Beginning Section 3")
    hypo_filtered_genotypes, hypo_significant_genes_and_eQTLs = HW3.HW3_module(genotypes, hypo_processed_data, mgi, "Hypothalamus")
    liver_filtered_genotypes, liver_significant_genes_and_eQTLs = HW3.HW3_module(genotypes, liver_processed_data, mgi, "Liver")

    ### Section 4 ###
    print("Beginning Section 4")
    regression_model_for_all_phenotypes = calculate_qtls(genotypes, PHENOTYPE_PATH)
    all_manhattan_plots(regression_model_for_all_phenotypes)
    regression_model_for_all_phenotypes = remove_insignificant(regression_model_for_all_phenotypes)

    ### Section 5 ###
    print("Beginning Section 5")
    hypo_close_columns = hypo_significant_genes_and_eQTLs['gene chromosome'] == hypo_significant_genes_and_eQTLs['snp chromosome']
    hypo_significant_genes_and_eQTLs = hypo_significant_genes_and_eQTLs[hypo_close_columns]
    liver_close_columns = liver_significant_genes_and_eQTLs['gene chromosome'] == liver_significant_genes_and_eQTLs['snp chromosome']
    liver_significant_genes_and_eQTLs = liver_significant_genes_and_eQTLs[liver_close_columns]
    combined_hypo_data = hypo_significant_genes_and_eQTLs.merge(hypo_filtered_genotypes, how='inner', right_index=True,
                                                                left_on='snp name')
    combined_liver_data = liver_significant_genes_and_eQTLs.merge(liver_filtered_genotypes, how='inner',
                                                                  right_index=True, left_on='snp name')
    combined_hypo_data_compressed = combined_hypo_data[['Locus', 'gene name', 'snp name']]
    combined_liver_data_compressed = combined_liver_data[['Locus', 'gene name', 'snp name']]
    compressed = [combined_hypo_data_compressed, combined_liver_data_compressed]
    triplets_datasets = []
    for combined_data in compressed:
        df = pd.DataFrame()
        for phenotype in regression_model_for_all_phenotypes.keys():
            phe = regression_model_for_all_phenotypes[phenotype]
            df = pd.concat([df, combined_data.merge(phe, how='inner', on='Locus')])
        df['phenotype name'] = df['phenotype'].map(REVERSE_PHENOTYPES)
        triplets_datasets.append(df[['Locus', 'gene name', 'phenotype']])

    ### Section 6 ###
    print("Beginning Section 6")
    hypo_triplets = [tuple(triplets_datasets[0].iloc[i]) for i in range(len(triplets_datasets[0]))]
    liver_triplets = [tuple(triplets_datasets[1].iloc[i]) for i in range(len(triplets_datasets[1]))]
    hypo_cause_test = ca.causality_model(hypo_triplets, hypo_processed_data)
    liver_cause_test = ca.causality_model(liver_triplets, liver_processed_data)
    print("Hypothalamus causality Analysis: \n", hypo_cause_test)
    print("Liver causality Analysis: \n", liver_cause_test)
