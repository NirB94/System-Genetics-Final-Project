import pandas as pd
import numpy as np
from pandas import DataFrame
from scipy.stats import f
import matplotlib.pyplot as plt
import seaborn as sns




#utils


def create_phenotype_df(file_path, phenotype_id):
    phenotype = pd.read_excel(file_path)
    phenotype = phenotype.loc[phenotype_id].dropna(axis=0)
    return phenotype

#question 2a


def regression_model(phenotype_id, phenotype_file, genotype_file, alpha=0.05):
    phenotype_df = create_phenotype_df(phenotype_file, phenotype_id)
    phenotype_index = phenotype_df.index[5:]
    genotype_df = pd.read_excel(genotype_file, skiprows=1)
    genotype_snps = genotype_df[genotype_df.columns[genotype_df.columns.isin(phenotype_index)]]
    phenotype_df = phenotype_df.loc[phenotype_index]
    p_vals = np.array([calculate_regression(genotype_snps.loc[i], phenotype_df, alpha) for i in range(genotype_snps.shape[0])])
    p_vals_minus_log = -np.log10(p_vals)
    results_df = pd.DataFrame({'Locus': genotype_df["Locus"], 'Chr': genotype_df["Chr_Build37"], 'P_value': p_vals, '-log_P_value': p_vals_minus_log})
    #results_df.to_excel("-log_p_value_results.xls")
    return results_df


# In[88]:


def calculate_regression(genotype_snp, phenotype_df, alpha):
    non_others_indexes = genotype_snp[(((genotype_snp == 'B') | (genotype_snp == 'D')) | (genotype_snp == 'H'))].index
    genotypes, phenotypes = genotype_snp[non_others_indexes], phenotype_df[non_others_indexes]
    genotypes = genotypes.map({'D': 0, 'H': 1, 'B':2})
    x, y = np.array(genotypes), np.array(phenotypes)
    avg_x, avg_y = np.average(x), np.average(y)
    n = y.shape[0]
    xi_yi_sum = np.sum(x * y)
    yi_square_sum = np.sum(y ** 2)
    xi_square_sum = np.sum(x ** 2)
    b1_hat = (xi_yi_sum - n * avg_x * avg_y)/(xi_square_sum - n * avg_x ** 2)
    b0_hat = avg_y - b1_hat * avg_x
    y_hat = b0_hat + b1_hat * x
    SSR = np.sum((y_hat - avg_y) ** 2)
    SST = np.sum((y-avg_y) ** 2)
    SSE = SST - SSR
    MSR = SSR #k == 2, therefore MSR=SSR
    MSE = (SSE/(n-2))
    F = f.ppf(q=1 - alpha, dfn=1, dfd=n - 2)
    F_star = MSR/MSE
    p_val = f.sf(F_star, dfn=1, dfd=n-2)
    return p_val


#question 2b


def create_manhattan_plot(df):
    df = df.sort_values('Chr')
    df.reset_index(inplace=True, drop=True)
    df['i'] = df.index
    plot = sns.relplot(data=df, x='i', y='-log_P_value', aspect=3.5, hue='Chr', palette='bright')
    chr_df = df.groupby('Chr')['i'].median()
    plot.ax.set_xlabel('Chromosome Number')
    plot.ax.set_xticks(chr_df)
    plot.ax.set_xticklabels(chr_df.index)
    plot.fig.suptitle('Manhattan plot')
    plt.show()



def best_score_snp(df):
    best_snp = df[df['-log_P_value'] == max(df['-log_P_value'])]
    print(best_snp)

def main():
    p_vals_log10 = regression_model(1434, "./phenotypes.xls", "./genotypes.xls")
    create_manhattan_plot(p_vals_log10)
    best_score_snp(p_vals_log10)


# In[125]:


if __name__ == '__main__':
    main()