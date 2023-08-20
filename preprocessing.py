import GEOparse
import numpy as np
import pandas as pd
import sklearn.preprocessing as skpp
import warnings

warnings.simplefilter("ignore")


def process_GEO(accession, tissue):
    """This function receives an accession identifier and a tissue name.
    The function then reads the corresponding GEO file.
    The function then creates a DataFrame and after some manipulations and arrangements,
    The function returns that said DataFrame."""
    gene_ID = 'GENE_SYMBOL' if tissue == 'Liver' else 'gene_assignment'
    gene_expression = GEOparse.get_GEO(accession)
    gpl = gene_expression.gpls.popitem()[1]
    geo_table = assign_tables(gene_expression.gsms.items(), tissue).merge(gpl.table[['ID', gene_ID]], how='left',
                                                                          left_on='ID_REF', right_on='ID')
    geo_table = geo_table.pivot_table(index=gene_ID, columns='sample_strain', values='VALUE')
    if tissue == 'Hypothalamus':
        geo_table = geo_table[geo_table.index != '---']
        geo_table.index = geo_table.index.map(lambda x: x.split(' // ')[1])
        geo_table = geo_table.groupby(gene_ID).mean()
        geo_table.rename(columns={'BXD92A': 'BXD92'}, inplace=True)
    else:
        geo_table['BXD11'] = geo_table[['BXD11', 'BXD11TY']].mean(axis=1)
        geo_table.drop('BXD11TY', inplace=True, axis=1)
    return geo_table


def assign_tables(gsms_items, tissue):
    """This is a helper function for process_GEO.
    The function receives a dict_items object and a tissue name.
    The function concat the tables created for each sample in the gsms_items object.
    The function filters out all samples that are not 'BXD'.
    The function returns the combined tables as a single DataFrame."""
    tables_list = []
    for sample_id, sample in gsms_items:
        strain_name = extract_strain_name(sample.metadata['characteristics_ch1'], tissue)
        if 'BXD' in strain_name:
            tables_list.append(sample.table.assign(sample=sample_id,
                                                   sample_strain=strain_name))
    return pd.concat(tables_list)


def extract_strain_name(char_list, tissue):
    """This is a helper function for assign_tables.
    The function receives a characteristics list and a tissue name.
    The function then extracts the 'BXD', and returns it."""
    booli = 0 if tissue == 'Liver' else 1
    return char_list[booli].split(' ')[1]


def normalize_hypo(hypo_df):
    """This function is used to normalize Hypothalamus Database into a standard normal distribution."""
    columns = hypo_df.columns
    hypo_df = hypo_df.applymap(np.log1p)
    normalized = skpp.StandardScaler().fit_transform(hypo_df[columns])
    hypo_df[columns] = normalized
    return hypo_df


def filter_genes(genes_table, slice_amount=0.5, final_amount=1500):
    """This function is used to take half of genes with maximal expression and 1500 genes with the highest variance.
    The function receives a DataFrame of genes, and defaulted slice_amount (0.5) and final_amount (1500).
    The function sorts the max and var Series in descending order.
    The function then takes half of the genes with the maximal expression, and 1500 genes with the highest variance.
    The function returns a DataFrame of these 1500 genes."""
    max_per_gene = genes_table.max(axis=1)
    var_per_gene = genes_table.var(axis=1)
    max_per_gene.sort_values(ascending=False, inplace=True)
    var_per_gene.sort_values(ascending=False, inplace=True)
    selected_genes = max_per_gene[:int(max_per_gene.count() * slice_amount)]
    print(f"Number of genes after filtering by maximal expression: {len(selected_genes)}")
    selected_genes = var_per_gene[selected_genes.index].sort_values(ascending=False)[:final_amount]
    print(f"Number of genes after filtering by variance of expression: {final_amount}")
    return genes_table.loc[selected_genes.index, :]


def run_preprocessing(accession, liver=True):
    """This function is used in the main module for running this whole module.
    The function receives an accession ID and a boolean variable called 'liver' defaulted to True.
    The functino runs the whole preprocessing module.
    The function saves the processed DataFrame, and returns it."""
    tissue = 'Liver' if liver else 'Hypothalamus'
    geo_table = process_GEO(accession, tissue)
    if not liver:
        geo_table = normalize_hypo(geo_table)
    geo_table = filter_genes(geo_table)
    geo_table.to_csv(f"{tissue}_data_after_processing.csv")
    return geo_table
