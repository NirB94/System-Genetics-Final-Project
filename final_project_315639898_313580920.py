import GEOparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from statsmodels.stats.multitest import fdrcorrection
import preprocessing as pp
import HW3
import regression
import time

GENOTYPES = 'genotypes.xls'
PHENOTYPE_PATH = 'phenotypes.xls'
MGI = 'MGI_Coordinates.Build37.rpt.txt'
PHENOTYPES = {"Cocaine response (10 mg/kg ip), locomotion from 0-15 min after first injection in an activity chamber "
               "for males [cm]": 1109, "Cocaine response (10 mg/kg ip), locomotion from 15-30 min after first"
                                         " injection in an activity chamber for males [cm]": 1110,
              "Cocaine response (10 mg/kg ip), locomotion"
               " from 30-45 min after first injection in an activity chamber for males [cm]": 1111}


### REGRESSION ###

def calculate_qtls(genotypes, phenotypes_path):
    p_vals_per_phenotype = dict()
    for name, ID in PHENOTYPES.items():
        phenotype = regression.create_phenotype_df(phenotypes_path, ID)
        filtered_genotypes = HW3.filtering(genotypes, extract_strains(phenotype))
        p_vals = regression.regression_model(phenotype, filtered_genotypes, name)
        p_vals['phenotype'] = ID
        p_vals_per_phenotype[name] = p_vals

    correction_df = pd.concat([p_vals_per_phenotype[key] for key in p_vals_per_phenotype.keys()])
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
        temp_df = temp_df[temp_df['P_value'] <= 0.05]
        regression_dict[phenotype] = temp_df
    return regression_dict



if __name__ == '__main__':
    t0 = time.time()
    liver_accession = 'GSE17522'
    hypothalamus_accession = 'GSE36674'
    genotypes = pd.read_excel(GENOTYPES, skiprows=1)
    mgi = pd.read_csv(MGI, sep='\t', header=0)

    ### Section 2 ###
    hypo_processed_data = pp.run_preprocessing(hypothalamus_accession, liver=False)
    liver_processed_data = pp.run_preprocessing(liver_accession)

    ### Section 3 ###
    hypo_significant_eQTLs = HW3.HW3_module(genotypes, hypo_processed_data, mgi, "Hypothalamus")
    liver_significant_eQTLs = HW3.HW3_module(genotypes, liver_processed_data, mgi, "Liver")

    ### Section 4 ###
    regression_model_for_all_phenotypes = calculate_qtls(genotypes, PHENOTYPE_PATH)
    all_manhattan_plots(regression_model_for_all_phenotypes)
    regression_model_for_all_phenotypes = remove_insignificant(regression_model_for_all_phenotypes)

    ### Section 4 remainings:
    ### Bug in calculate_qtls: filtered genotypes returns empty and there's no 'Locus' column to use!
    ### Need to figure out how to overcome this.
    ### Then - Build Triplets (For section 5)


    t1 = time.time()
    print("Time took:\n", (t1-t0)/60, "minutes")
