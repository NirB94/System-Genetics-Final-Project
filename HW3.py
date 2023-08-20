from copy import deepcopy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.stats.multitest import fdrcorrection
from scipy.stats import linregress as lrs
import seaborn as sns


def filtering(genotypes, strains):
    relevant_strains = [column for column in genotypes.columns if column in strains]
    columns = list(genotypes.columns)[:3]
    columns.extend(relevant_strains)
    final_genotypes = genotypes[columns]
    drop_indices = []
    for i in range(len(final_genotypes) - 1):
        if (final_genotypes.iloc[i, 3:]).equals(final_genotypes.iloc[i + 1, 3:]):
            drop_indices.append(i + 1)
    final_genotypes = final_genotypes.drop(drop_indices)
    return final_genotypes


def mean_data(table):
    cols_set = {col.split('_')[0] for col in table.columns if 'BXD' in col}
    means_table = pd.DataFrame({'data': table['data']})
    for strain in cols_set:
        temp_cols = [col for col in table.columns if col.split('_')[0] == strain]
        means_table[strain] = table[temp_cols].mean(axis=1)
    sorted_cols = list(table.columns[:1])
    sorted_cols.extend(sorted(means_table.columns[1:], key=lambda x: int(x[3:])))
    means_table = means_table[sorted_cols]
    return means_table


def association_model(genotypes_df, expression_df, alpha=0.05):
    relevant_cols = [col for col in genotypes_df.columns if col in expression_df.columns]
    genotypes_df = genotypes_df[genotypes_df.columns[:3].to_list() + relevant_cols]
    strains_dict = dict()
    for i in range(len(expression_df)):
        p_vals_list = []
        phe = expression_df.iloc[i]
        for index in genotypes_df.index:
            gen = genotypes_df.loc[index]
            non_others_indexes = gen[(((gen == 'B') | (gen == 'D')) | (gen == 'H'))].index
            gen, phe = gen[non_others_indexes], phe[non_others_indexes].astype(float)
            gen = gen.map({'B': 2, 'H': 1, 'D': 0}).astype(int)
            p_vals_list.append(lrs(gen, phe)[3])
        strains_dict[phe['data']] = p_vals_list
    raw_mat = pd.DataFrame(strains_dict)
    raw_mat = raw_mat.T
    raw_mat.index = expression_df['data']
    raw_mat.columns = genotypes_df.index
    stacked = raw_mat.stack()
    stacked = pd.Series(fdrcorrection(stacked, 0.05)[1], index=stacked.index)
    final_mat = stacked.unstack()
    final_mat.reset_index(inplace=True)
    final_mat['min_p_val'] = final_mat.min(axis=1)
    final_mat = final_mat[final_mat['min_p_val'] < alpha]
    final_mat.drop('min_p_val', axis='columns', inplace=True)
    final_mat.set_index('data', inplace=True)
    return final_mat


def create_gene_map_loc(mgi_df, p_vals_df):
    mgi_df = deepcopy(mgi_df)
    mgi_df.index = mgi_df['marker symbol']
    mgi_df = mgi_df[['representative genome chromosome', 'representative genome start', 'representative genome end']]
    relevant_gene_loc = mgi_df[mgi_df.index.isin(p_vals_df.index)]
    relevant_gene_loc.columns = ['chromosome', 'start', 'end']
    mgi_df = mgi_df.dropna(inplace=False)
    return relevant_gene_loc, mgi_df


def cis_trans(snp, gene):
    if int(gene['chromosome']) == snp['Chr_Build37']:
        position_snp = snp['Build37_position']
        if np.abs(position_snp - gene['start']) <= 2000000 or np.abs(position_snp - gene['end']) <= 2000000:
            return 'cis'
        else:
            return 'trans'
    return 'trans'


def all_loci(gene, genotypes_df):
    d = {}
    for i in range(len(genotypes_df)):
        d[i] = cis_trans(genotypes_df.iloc[i], gene)
    return pd.Series(d)


def cis_trans_for_all_loci(genes, genotypes_df, every_loci):
    d = {}
    for i in range(len(genes)):
        d[genes.iloc[i].name] = (all_loci(genes.iloc[i], genotypes_df))
    d = pd.DataFrame(d).T
    d.index.rename('marker symbol', inplace=True)
    d = d.rename({i: every_loci[i] for i in range(len(list(d.columns)))}, axis='columns')
    return d


def count_significant_cis_trans(p_values, cis_trans_s):
    combined = pd.DataFrame(p_values.unstack().rename('p_values')).join(
        cis_trans_s.rename_axis('data').unstack().rename('cis-trans'))
    counts_df = pd.DataFrame(combined.loc[combined.p_values <= 0.05, 'cis-trans']).reset_index()
    counts_df.rename({'level_0': 'snp'}, axis=1, inplace=True)
    val_counts = counts_df['cis-trans'].value_counts()
    counts_dict = {'Significant': val_counts['trans'] + val_counts['cis'], 'Cis': val_counts['cis'],
                   'Trans': val_counts['trans']}
    return counts_df, counts_dict


def eQTL_association_df(p_values, filtered_genotypes_df):
    p_values2 = p_values.reset_index()
    p_values2 = p_values2.drop('data', axis=1)
    sum_of_eQTL = (p_values2 < 0.05).sum()
    eQTL_df = pd.DataFrame({'Count': sum_of_eQTL})
    filtered_genotypes_df['Chr_Build37'].index = eQTL_df.index
    return pd.DataFrame(eQTL_df).join(filtered_genotypes_df['Chr_Build37'])


def plot_eQTL(eQTL_df, data_name):
    eQTL_df['place'] = range(len(eQTL_df))
    eQTL_sorted = eQTL_df.sort_values('Chr_Build37')
    eQTL_sorted.groupby('Chr_Build37')
    eQTL_df.reset_index(inplace=True, drop=True)
    eQTL_df['index'] = eQTL_df.index
    sns.set(font_scale=1.25)
    sns.set_style('dark')
    plot = sns.relplot(data=eQTL_df, x='index', y='Count', aspect=3.7, hue='Chr_Build37', palette='dark', legend=None,
                       s=200)
    chrom_df = eQTL_df.groupby('Chr_Build37')['index'].median()
    plot.ax.set_xlabel('Chromosome No.')
    plot.ax.set_xticks(chrom_df)
    plot.ax.set_xticklabels(chrom_df.index)
    plot.fig.suptitle('Number of genes associated with each eQTL')
    plot.set(yticks=[i for i in range(0, 4)])
    plt.savefig(f'Number of genes associated with each eQTL in {data_name}.png')
    plt.show()


def create_eQTL_ditribution_data(p_values, cis_trans):
    df = pd.DataFrame(p_values.unstack().rename("p_values")).join(
        cis_trans.rename_axis("data").unstack().rename("cis_trans"))
    filter_arr = lambda x: df.loc[df['cis_trans'] == x, 'p_values'].to_numpy()
    cis_arr, trans_arr = filter_arr('cis'), filter_arr('trans')
    log_cis_arr, log_trans_arr = -np.log10(cis_arr), -np.log10(trans_arr)
    significant_cis_arr, significant_trans_arr = cis_arr[cis_arr <= 0.05], trans_arr[trans_arr <= 0.05]
    log_significant_cis_arr, log_significant_trans_arr = -np.log10(significant_cis_arr), -np.log10(
        significant_trans_arr)
    return log_significant_cis_arr, log_significant_trans_arr, log_cis_arr, log_trans_arr


def plot_distribution_graph(cis, trans, sig, data_name):
    cis, trans = np.array(list(cis)), np.array(list(trans))
    g = sns.kdeplot(cis, color="r", label='Cis')
    sns.kdeplot(trans, color="b", label='Trans', bw=0.3)
    plt.legend(loc='upper right')
    g.set_ylabel('Density')
    g.set_xlabel('-log p value')
    plt.title(f"Distribution of {sig} eQTLs P-values in {data_name}")
    plt.savefig(f"Distribution of {sig} eQTLs P-values in {data_name}")
    plt.show()


def create_snp_gene_df(sig_df, genotypes_df, relevant_gene_map_loc_df):
    df = relevant_gene_map_loc_df.merge(sig_df, right_on='data', left_index=True)
    df['gene location'] = (df['start'] + df['end']) / 2
    df.drop(['start', 'end'], axis=1, inplace=True)
    df['snp'] = df['snp'].astype(int)
    df = df.merge(genotypes_df[['Chr_Build37', 'Build37_position']], left_on='snp', right_index=True)
    columns = ['gene name', 'gene location', 'gene chromosome', 'snp name', 'snp location', 'snp chromosome',
               'cis-trans']
    df.rename({'data': 'gene name', 'chromosome': 'gene chromosome', 'Chr_Build37': 'snp chromosome',
               'Build37_position': 'snp location', 'snp': 'snp name'}, axis=1, inplace=True)
    df = df[columns]
    df['gene chromosome'] = df['gene chromosome'].astype(int)
    return df


def find_max_pos(chr_data):
    ser = chr_data.groupby('representative genome chromosome')['representative genome end'].max()
    for i in range(2, len(ser) + 1):
        ser[i] += ser[i - 1]
    return ser


def reg_location(df, max_pos_ser, is_gene=True):
    form = 'gene' if is_gene else 'snp'
    for i in range(len(df)):
        df.loc[i, f'normalized location of {form}'] = calc_reg_loc(df.iloc[i], max_pos_ser, form)
    return df


def calc_reg_loc(gene, max_pos_ser, form):
    if gene[f'{form} chromosome'] == 1:
        return gene[f'{form} location']
    else:
        return gene[f'{form} location'] + max_pos_ser[gene[f'{form} chromosome'] - 1]


def plot_cis_trans_gene_position(chr_df, max_pos_ser, data_name):
    chr_no = list(range(1, 21))
    fig, ax = plt.subplots()
    cis = chr_df.loc[chr_df['cis-trans'] == 'cis']
    x = cis['normalized location of snp']
    y = cis['normalized location of gene']
    cis = ax.plot(x, y, 'o', label='cis', markersize=8)
    trans = chr_df.loc[chr_df['cis-trans'] == 'trans']
    x = trans['normalized location of snp']
    y = trans['normalized location of gene']
    trans = ax.plot(x, y, 'o', label='trans', markersize=3)
    ax.set_ylabel('Gene Location')
    ax.set_xlabel('SNP Location')
    ax.legend()
    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='y', labelsize=9)
    ax.set_xticks(max_pos_ser)
    plt.xticks(max_pos_ser, chr_no)
    ax.set_xticks(max_pos_ser)
    plt.yticks(max_pos_ser, chr_no)
    plt.title(f'Visualization of cis and trans genes and SNPs locations in {data_name}')
    plt.savefig(f'Visualization of cis and trans genes and SNPs locations in {data_name}.png')
    plt.show()


def HW3_module(genotypes_file, geo_data, mgi_file, data_name):
    data_name = f'{data_name} Dataset'
    strains = mean_data(geo_data)
    genotypes = filtering(genotypes_file, strains.columns)

    p_vals = association_model(genotypes, strains)  ## Takes a long time, csv attached and imported in the line below:
    p_vals.to_csv(f'{data_name} Association Model.csv')
    # p_vals = pd.read_csv("after association test.csv", index_col=[0])

    loci = p_vals.columns

    relevant, loc2 = create_gene_map_loc(mgi_file, p_vals)

    cis_trans_df = cis_trans_for_all_loci(relevant, genotypes, loci)

    ### Question 1 ###
    counts = count_significant_cis_trans(p_vals, cis_trans_df)
    sig = counts[0]
    counts_val = counts[1]
    print(f"Significant, cis and trans counts in {data_name}:\n", counts_val)

    ### Question 2 ###
    eQTL = eQTL_association_df(p_vals, genotypes)
    plot_eQTL(eQTL, data_name)

    ### Question 3 ###
    sig_cis, sig_trans, cis, trans = create_eQTL_ditribution_data(p_vals, cis_trans_df)
    plot_distribution_graph(cis, trans, "all genome", data_name)
    plot_distribution_graph(sig_cis, sig_trans, "significant", data_name)

    ### Question 4 ###
    gene_snp_locations = create_snp_gene_df(sig, genotypes, relevant)
    chr_data = loc2[
        (loc2['representative genome chromosome'] != 'Y') & (loc2['representative genome chromosome'] != 'MT')]
    chr_data['representative genome chromosome'] = chr_data['representative genome chromosome'].astype(float)
    max_pos = find_max_pos(chr_data)
    gene_snp_locations = reg_location(gene_snp_locations, max_pos)
    gene_snp_locations = reg_location(gene_snp_locations, max_pos, is_gene=False)
    plot_cis_trans_gene_position(gene_snp_locations, max_pos, data_name)
    return gene_snp_locations['snp name']
