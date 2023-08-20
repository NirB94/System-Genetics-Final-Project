import GEOparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import preprocessing as pp
import HW3
import regression

GENOTYPES = 'genotypes.xls'
MGI = 'MGI_Coordinates.Build37.rpt.txt'
PHENOTYPES = [("Cocaine response (10 mg/kg ip), locomotion from 0-15 min after first injection in an activity chamber "
               "for males [cm]", 1109), ("Cocaine response (10 mg/kg ip), locomotion from 15-30 min after first"
                                         " injection in an activity chamber for males [cm]", 1110),
              ("Cocaine response (10 mg/kg ip), locomotion"
               " from 30-45 min after first injection in an activity chamber for males [cm]", 1111)]











if __name__ == '__main__':
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


