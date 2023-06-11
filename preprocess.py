import sys
import os
import numpy as np
import torch
import torch.utils.data as du
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import pandas as pd
import candle
import time
import logging
import networkx as nx
import networkx.algorithms.components.connected as nxacc
import networkx.algorithms.dag as nxadag
from pathlib import Path
import improve_utils
import candle
import urllib
import urllib.request
from sklearn.model_selection import train_test_split
import requests



file_path = os.path.dirname(os.path.realpath(__file__))
additional_definitions = None
required = None

CANDLE_DATA_DIR=os.getenv("CANDLE_DATA_DIR")
proxies={
"HTTP_PROXY":"http://proxy.alcf.anl.gov:3128",
"HTTPS_PROXY":"http://proxy.alcf.anl.gov:3128",
"http_proxy":"http://proxy.alcf.anl.gov:3128",
"https_proxy":"http://proxy.alcf.anl.gov:3128"
}


def get_drug_response_data(df, metric):
    
    # df = rs_train.copy()
    smiles_df = improve_utils.load_smiles_data()
    data_smiles_df = pd.merge(df, smiles_df, on = "improve_chem_id", how='left') 
    data_smiles_df = data_smiles_df.dropna(subset=[metric])
    data_smiles_df = data_smiles_df[['improve_sample_id', 'smiles', metric]]
    data_smiles_df = data_smiles_df.drop_duplicates()
    data_smiles_df = data_smiles_df.reset_index(drop=True)

    return data_smiles_df





def preprocess_ccle(opt):

    data_path = os.path.join(CANDLE_DATA_DIR, opt['model_name'], 'Data')

    get_data(data_url=opt['data_url'], cache_subdir=os.path.join(data_path, 'dc_original'), download=opt['download_data'], svn=False)
    
    csa_data_folder = os.path.join(CANDLE_DATA_DIR, opt['model_name'], 'Data', 'csa_data', 'raw_data')
    splits_dir = os.path.join(csa_data_folder, 'splits') 
    x_data_dir = os.path.join(csa_data_folder, 'x_data')
    y_data_dir = os.path.join(csa_data_folder, 'y_data')

    if not os.path.exists(csa_data_folder) and opt['download_data']:
        print('creating folder: %s'%csa_data_folder)
        os.makedirs(csa_data_folder)
        os.mkdir( splits_dir  )
        os.mkdir( x_data_dir  )
        os.mkdir( y_data_dir  )
    
        print('downloading data')
        for file in ['CCLE_all.txt', 'CCLE_split_0_test.txt', 'CCLE_split_0_train.txt', 'CCLE_split_0_val.txt']:
            # url = f'https://ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/single_drug_drp/csa_data/splits/{file}'
            # save_request_file(url=url, save_loc=splits_dir+f'/{file}')
            urllib.request.urlretrieve(f'https://ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-imp-2023/csa_data/splits/{file}',
            splits_dir+f'/{file}')

        for file in ['cancer_mutation_count.txt', 'drug_SMILES.txt','drug_ecfp4_512bit.txt' ]:
            # url=f'https://ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/single_drug_drp/csa_data/x_data/{file}'
            # save_request_file(url=url, save_loc=x_data_dir+f'/{file}')
            urllib.request.urlretrieve(f'https://ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-imp-2023/csa_data/x_data/{file}',
            x_data_dir+f'/{file}')

        for file in ['response.txt']:
            urllib.request.urlretrieve(f'https://ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-imp-2023/csa_data/y_data/{file}',
            y_data_dir+f'/{file}')




    if opt['data_type']=='ccle_candle':
        data_type = 'CCLE'   #'CCLE'
    metric = opt['metric']


    train_file =  os.path.join(data_path, opt['train']) #"drugcell_train.txt"
    val_file =  os.path.join(data_path,opt['test'])   #"drugcell_val.txt"
    test_file =  os.path.join(data_path,opt['infer'] )  #"drugcell_test.txt"


    drug_index_out = os.path.join(data_path,opt['drug2id'])
    cell_index_out = os.path.join(data_path,opt['cell2id'])
    gene_index_out = os.path.join(data_path,opt['gene2id'])
    cell_mutation_out = os.path.join(data_path,opt['genotype'])
    drug_fingerprint_out = os.path.join(data_path,opt['fingerprint'])


    rs_all = improve_utils.load_single_drug_response_data(source=data_type, split=0,
                                                        split_type=["train", "test", 'val'],
                                                        y_col_name=metric)

    rs_train = improve_utils.load_single_drug_response_data(source=data_type,
                                                            split=0, split_type=["train"],
                                                            y_col_name=metric)
    rs_test = improve_utils.load_single_drug_response_data(source=data_type,
                                                        split=0,
                                                        split_type=["test"],
                                                        y_col_name=metric)
    rs_val = improve_utils.load_single_drug_response_data(source=data_type,
                                                        split=0,
                                                        split_type=["val"],
                                                        y_col_name=metric)


    train_df = get_drug_response_data(rs_train, metric)
    test_df = get_drug_response_data(rs_test, metric)
    val_df = get_drug_response_data(rs_val, metric)
    all_df = pd.concat([train_df, val_df, test_df], axis=0)
    all_df = all_df.sort_values(by='improve_sample_id')
    all_df.reset_index(drop=True, inplace=True)

    if opt['data_split_seed']>-1:
        train_df, val_df = train_test_split(all_df, test_size=0.2, random_state = opt['data_split_seed'])
        test_df, val_df = train_test_split(val_df, test_size=0.5, random_state = opt['data_split_seed'])
        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
        val_df.reset_index(drop=True, inplace=True)

    train_df.to_csv(train_file, header=None, index=None, sep ='\t')
    test_df.to_csv(test_file, header=None, index=None, sep ='\t')
    val_df.to_csv(val_file, header=None, index=None, sep ='\t')



    #gene index file
    print("loading mutation data...")
    mutation_data = improve_utils.load_cell_mutation_data(gene_system_identifier="Entrez")
    mutation_data = mutation_data.reset_index()
    gene_data = mutation_data.columns

    gene_data = gene_data[1:]
    gene_list = list(set(list(gene_data)))
    gene_df = pd.DataFrame(gene_data)
    # saving the genes
    gene_df.to_csv(gene_index_out, sep='\t', header=None)


    data_df = rs_all.copy()
    #improve id
    improve_data_list = list(set(data_df.improve_sample_id.tolist()))

    mutation_data = mutation_data.drop_duplicates(subset=['improve_sample_id'])
    assert mutation_data.index.nunique() == mutation_data.shape[0]

    cell2ind = mutation_data[['improve_sample_id']]

    cell2mut_df = mutation_data.drop(columns=['improve_sample_id'])
    cell2mut_df.to_csv(cell_mutation_out, header=None, index=None)

    cell2ind.to_csv(cell_index_out, sep='\t', header=None)

    #drug2ind
    se = improve_utils.load_smiles_data()
    data_smiles = pd.merge(data_df, se, on = 'improve_chem_id', how='left')
    drug_id_df = data_smiles.drop_duplicates(subset=['smiles'])
    drug_id_df.reset_index(drop=True, inplace=True)

    drug2ind = drug_id_df[['smiles']]
    drug2ind.to_csv(drug_index_out, sep='\t', header=None)


    fp = improve_utils.load_morgan_fingerprint_data()
    fp_df = fp.loc[drug_id_df.improve_chem_id, :]
    fp_df.to_csv(drug_fingerprint_out, index=None, header=None)

    onto_in = os.path.join(data_path, 'dc_original', opt['onto'])
    onto_out = os.path.join(data_path, opt['onto'])

    print('onto in', onto_in, 'onto out', onto_out)
    create_ont(onto_in, onto_out, gene_list) 

    return gene_list, train_df, test_df, val_df

def create_ont(ont_in, ont_out, gene_list):
    ont_df = pd.read_csv(ont_in, sep='\t', header=None)
    ont_default_df = ont_df[ont_df[2] == 'default']
    ont_gene_df = ont_df[ont_df[2] == 'gene']
    ont_gene_df = ont_gene_df[ont_df[1].isin(gene_list)]
    GO_list = list(set(ont_gene_df[0].tolist()))
    ont_default_df = ont_default_df[(ont_default_df[0].isin(GO_list)) | (ont_default_df[1].isin(GO_list))]
    ont_cat_df = pd.concat([ont_default_df, ont_gene_df])
    ont_cat_df.to_csv(ont_out, sep='\t', index=None, header=None)



def get_data(data_url, cache_subdir, download=True, svn=False):
    print('downloading data')
    # cache_subdir = os.path.join(CANDLE_DATA_DIR, 'SWnet', 'Data')
    
    if download and svn:
        os.makedirs(cache_subdir, exist_ok=True)
        os.system(f'svn checkout {data_url} {cache_subdir}')   
        print('downloading done') 
    elif download and svn==False:
        os.makedirs(cache_subdir, exist_ok=True)
        # urllib.request.urlretrieve('https://raw.githubusercontent.com/idekerlab/DrugCell/public/data/cell2ind.txt', f'{cache_subdir}/cell2ind.txt')
        urllib.request.urlretrieve('https://raw.githubusercontent.com/idekerlab/DrugCell/public/data/drugcell_ont.txt', f'{cache_subdir}/drugcell_ont.txt')


def download_ccle_data(opt):

    data_path = os.path.join(CANDLE_DATA_DIR, opt['model_name'], 'Data')
    get_data(data_url=opt['data_url'], cache_subdir=os.path.join(data_path, 'dc_original'), download=True, svn=False)
    
    csa_data_folder = os.path.join(CANDLE_DATA_DIR, opt['model_name'], 'Data', 'csa_data', 'raw_data')
    splits_dir = os.path.join(csa_data_folder, 'splits') 
    x_data_dir = os.path.join(csa_data_folder, 'x_data')
    y_data_dir = os.path.join(csa_data_folder, 'y_data')

    if not os.path.exists(csa_data_folder):
        print('creating folder: %s'%csa_data_folder)
        os.makedirs(csa_data_folder)
        os.mkdir( splits_dir  )
        os.mkdir( x_data_dir  )
        os.mkdir( y_data_dir  )
    

        for file in ['CCLE_all.txt', 'CCLE_split_0_test.txt', 'CCLE_split_0_train.txt', 'CCLE_split_0_val.txt']:
            urllib.request.urlretrieve(f'https://ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-imp-2023/csa_data/splits/{file}',
            splits_dir+f'/{file}')

        for file in ['cancer_mutation_count.txt', 'drug_SMILES.txt','drug_ecfp4_512bit.txt' ]:
            urllib.request.urlretrieve(f'https://ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-imp-2023/csa_data/x_data/{file}',
            x_data_dir+f'/{file}')

        for file in ['response.txt']:
            urllib.request.urlretrieve(f'https://ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-imp-2023/csa_data/y_data/{file}',
            y_data_dir+f'/{file}')





