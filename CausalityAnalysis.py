import math
import pandas as pd
import numpy as np
import scipy.stats as st
from scipy.stats import linregress as lrs


GENOTYPES = pd.read_excel('genotypes.xls', skiprows=1).set_index('Locus').iloc[:, 3:]
PHENOTYPES = {'Morphine response (50 mg/kg ip), locomotion (open field) from 120-135 min after injection in an activity chamber for males [cm]': 964,
    "Morphine response (50 mg/kg ip), locomotion (open field) from 45-60 min after injection in an activity"
              " chamber for males [cm]": 970,
              "Morphine response (50 mg/kg ip), locomotion (open field) from 45-60 min after injection in an activity"
              " chamber for females [cm]": 1224,
              "Morphine response (50 mg/kg ip), locomotion (open field) from 45-60 min after injection in an activity"
              " chamber for males and females [cm]": 1478}
PHENOTYPES_FILE = pd.read_excel('phenotypes.xls')
SPECIFIC_PHENOTYPES = PHENOTYPES_FILE[PHENOTYPES_FILE['Phenotype'].isin(list(PHENOTYPES.keys()))].iloc[:, 7:].dropna(axis=1)


def get_data_for_triplet(triplet, ge_processed_data):
    """This function
    The function receives 2 parameters: a triplet of (genotype, gene, phenotype), and Gene expression processed data.
    The function filters genotype and takes homozygous strains only.
    The function then filters all relevant indices i.e. homozygous, and non NaN values.
    The function returns a DataFrame of the remaining indices in every member of the triplet.
    """
    phe = SPECIFIC_PHENOTYPES.loc[triplet[2]]
    ge = ge_processed_data.loc[triplet[1]]
    genotype = GENOTYPES.loc[triplet[0]]
    genotype = genotype[genotype.isin(('B', 'D'))]
    indices = sorted(list(set(genotype.index).intersection(set(phe.index)).intersection(set(ge.index))),
                     key=lambda x: int(x[3:]))  # intersection of 3 indices sets, sorted by number of BXD
    phe, ge, genotype = phe[indices], ge[indices], genotype[indices]
    genotype = genotype.map({'B': 1, 'D': 0})
    df = pd.DataFrame({"L": genotype, "R": ge, "C": phe})
    df.sort_values(by=['L'], inplace=True)
    return df


def calc_all_distributions(lrc_df):
    """This function receives a DataFrame, which is the output of get_data_for_triplet function.
    The function adds all distributions and probabilities needed as columns.
    The function calculates likelihood of all relevant models taught in class.
    The function returns the modified DataFrame."""
    lrc_df['N(R|L)'], lrc_df['N(C|L)'] = calc_parameters_given_L('R', lrc_df), calc_parameters_given_L('C', lrc_df)

    E_R, std_R = np.mean(lrc_df['R']), np.std(lrc_df['R'])
    E_C, std_C = np.mean(lrc_df['C']), np.std(lrc_df['C'])
    rho = lrs(lrc_df['C'].astype(float), lrc_df['R'].astype(float)).rvalue
    lrc_df['N(R|C)'] = lrc_df.apply(lambda row: calc_parameters(row, E_R, std_R, E_C, std_C, rho, 'C'), axis=1)
    lrc_df['N(C|R)'] = lrc_df.apply(lambda row: calc_parameters(row, E_C, std_C, E_R, std_R, rho, 'R'), axis=1)

    lrc_df['P(Ri|Li)'] = lrc_df.apply(lambda row: calc_probability(row['R'], row['N(R|L)']), axis=1)
    lrc_df['P(Ci|Li)'] = lrc_df.apply(lambda row: calc_probability(row['C'], row['N(C|L)']), axis=1)
    lrc_df['P(Ri|Ci)'] = lrc_df.apply(lambda row: calc_probability(row['R'], row['N(R|C)']), axis=1)
    lrc_df['P(Ci|Ri)'] = lrc_df.apply(lambda row: calc_probability(row['C'], row['N(C|R)']), axis=1)

    lrc_df['M1'] = lrc_df.apply(lambda row: calc_log_likelihood(row['P(Ri|Li)'], row['P(Ci|Ri)']), axis=1)
    lrc_df['M2'] = lrc_df.apply(lambda row: calc_log_likelihood(row['P(Ci|Li)'], row['P(Ri|Ci)']), axis=1)
    lrc_df['M3'] = lrc_df.apply(lambda row: calc_log_likelihood(row['P(Ri|Li)'], row['P(Ci|Li)']), axis=1)

    return lrc_df


def calc_parameters_given_L(col, df):
    return create_dist_lst(col, df, 0) + create_dist_lst(col, df, 1)


def create_dist_lst(col, df, val):
    vals = np.array(df.loc[df['L'] == val, col])
    dist = (np.mean(vals), math.pow(np.std(vals), 2))
    return [dist for i in range(vals.size)]


def calc_parameters(row, mean1, std1, mean2, std2, coef, col):
    return mean1 + ((coef * std1) / std2) * (row[col] - mean2), (std1 ** 2) * (1 - coef ** 2)


def calc_probability(col, dist):
    return st.norm.pdf(x=col, loc=dist[0], scale=math.sqrt(dist[1]))


def calc_log_likelihood(prob1, prob2):
    return math.log(0.5 * prob1 * prob2)


def calc_likelihood_ratio(df):
    m_likelihoods = [df[f'M{i}'].sum() for i in range(1, 4)]
    max_val = max(m_likelihoods)
    model = np.argmax(m_likelihoods) + 1
    m_likelihoods.remove(max_val)
    likelihood_ratio = max_val - max(m_likelihoods)
    return likelihood_ratio, model


def calc_likelihood_ratio_permuted(df, model):
    m_likelihoods = [df[f'M{i}'].sum() for i in range(1, 4)]
    max_val = m_likelihoods[model - 1]
    m_likelihoods.remove(max_val)
    return max_val - max(m_likelihoods)


def permutation_test(df, rounds):
    L_ratio, model = calc_likelihood_ratio(df)
    L_ratio_list = []
    for i in range(rounds):
        L = df['L']
        R = np.random.permutation(df['R'])
        C = np.random.permutation(df['C'])
        arr = np.array([L, R, C])
        new_df = pd.DataFrame(arr, index=['L', 'R', 'C']).T
        new_df = calc_all_distributions(new_df)
        L_ratio_permuted = calc_likelihood_ratio_permuted(new_df, model)
        L_ratio_list.append(L_ratio_permuted)
    p_val = len([LR for LR in L_ratio_list if LR >= L_ratio]) / rounds
    return model, p_val


def causality_model(triplets, processed_data, rounds=100):
    dic = {}
    for triplet in triplets:
        lrc = get_data_for_triplet(triplet, processed_data)
        lrc = calc_all_distributions(lrc)
        dic[triplet] = permutation_test(lrc, rounds)
    return dic
