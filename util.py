import sys
import torch
import networkx as nx
import networkx.algorithms.components.connected as nxacc
import networkx.algorithms.dag as nxadag
import numpy as np
import os
import torch.utils.data as du
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from drugcell_NN import *
import argparse
from scipy.stats import spearmanr

def pearson_corr(x, y):
	xx = x - torch.mean(x)
	yy = y - torch.mean(y)

	return torch.sum(xx*yy) / (torch.norm(xx, 2)*torch.norm(yy,2))

def load_ontology(file_name, gene2id_mapping):

	dG = nx.DiGraph()
	term_direct_gene_map = {}
	term_size_map = {}

	file_handle = open(file_name)

	gene_set = set()

	for line in file_handle:

		line = line.rstrip().split()
		
		if line[2] == 'default':
			dG.add_edge(line[0], line[1])
		else:
			if line[1] not in gene2id_mapping:
				continue

			if line[0] not in term_direct_gene_map:
				term_direct_gene_map[ line[0] ] = set()

			term_direct_gene_map[line[0]].add(gene2id_mapping[line[1]])

			gene_set.add(line[1])

	file_handle.close()

	print('There are', len(gene_set), 'genes')

	for term in dG.nodes():
		
		term_gene_set = set()

		if term in term_direct_gene_map:
			term_gene_set = term_direct_gene_map[term]

		deslist = nxadag.descendants(dG, term)

		for child in deslist:
			if child in term_direct_gene_map:
				term_gene_set = term_gene_set | term_direct_gene_map[child]

		# jisoo
		if len(term_gene_set) == 0:
			print('There is empty terms, please delete term:', term)
			sys.exit(1)
		else:
			term_size_map[term] = len(term_gene_set)

	leaves = [n for n in dG.nodes if dG.in_degree(n) == 0]
	#leaves = [n for n,d in dG.in_degree() if d==0]

	uG = dG.to_undirected()
	connected_subG_list = list(nxacc.connected_components(uG))

	print('There are', len(leaves), 'roots:', leaves[0])
	print('There are', len(dG.nodes()), 'terms')
	print('There are', len(connected_subG_list), 'connected componenets')

	if len(leaves) > 1:
		print('There are more than 1 root of ontology. Please use only one root.')
		sys.exit(1)
	if len(connected_subG_list) > 1:
		print( 'There are more than connected components. Please connect them.')
		sys.exit(1)

	return dG, leaves[0], term_size_map, term_direct_gene_map


def load_train_data(file_name, cell2id, drug2id, train_subset):
	feature = []
	label = []

	with open(file_name, 'r') as fi:
		for il, line in enumerate(fi):
			tokens = line.strip().split('\t')

			feature.append([cell2id[tokens[0]], drug2id[tokens[1]]])
			label.append([float(tokens[2])])
			if train_subset and il == train_subset:
				break
	return feature, label


# instead of reading train/test data from file, pass the dataframe
def load_train_data_from_df(file_name, cell2id, drug2id, train_subset):
	feature = []
	label = []

	with open(file_name, 'r') as fi:
		for il, line in enumerate(fi):
			tokens = line.strip().split('\t')

			feature.append([cell2id[tokens[0]], drug2id[tokens[1]]])
			label.append([float(tokens[2])])
			if train_subset and il == train_subset:
				break
	return feature, label


def prepare_predict_data(test_file, cell2id_mapping_file, drug2id_mapping_file, train_subset):

	# load mapping files
	cell2id_mapping = load_mapping(cell2id_mapping_file)
	drug2id_mapping = load_mapping(drug2id_mapping_file)

	test_feature, test_label = load_train_data(test_file, cell2id_mapping, drug2id_mapping, train_subset)

	print('Total number of cell lines = %d' % len(cell2id_mapping))
	print('Total number of drugs = %d' % len(drug2id_mapping))

	return (torch.Tensor(test_feature), torch.Tensor(test_label)), cell2id_mapping, drug2id_mapping


def load_mapping(mapping_file):

	mapping = {}

	file_handle = open(mapping_file)

	for line in file_handle:
		line = line.rstrip().split()
		mapping[line[1]] = int(line[0])

	file_handle.close()
	
	return mapping


def prepare_train_data(train_file, test_file, cell2id_mapping_file, drug2id_mapping_file, train_subset):

	# load mapping files
	cell2id_mapping = load_mapping(cell2id_mapping_file)
	drug2id_mapping = load_mapping(drug2id_mapping_file)

	train_feature, train_label = load_train_data(train_file, cell2id_mapping, drug2id_mapping, train_subset)
	test_feature, test_label = load_train_data(test_file, cell2id_mapping, drug2id_mapping, train_subset=None)

	print('Total number of cell lines = %d' % len(cell2id_mapping))
	print('Total number of drugs = %d' % len(drug2id_mapping))

	return (torch.Tensor(train_feature), torch.FloatTensor(train_label), torch.Tensor(test_feature), torch.FloatTensor(test_label)), cell2id_mapping, drug2id_mapping


def build_input_vector(input_data, cell_features, drug_features):
	genedim = len(cell_features[0,:])
	drugdim = len(drug_features[0,:])
	feature = np.zeros((input_data.size()[0], (genedim+drugdim)))

	for i in range(input_data.size()[0]):
		feature[i] = np.concatenate((cell_features[int(input_data[i,0])], drug_features[int(input_data[i,1])]), axis=None)

	feature = torch.from_numpy(feature).float()
	return feature



def predict_dcell(opt, data_path, infer_path):
	
	# predict = os.path.join(data_path, opt['infer'] )
	predict = infer_path
	
	gene2id = os.path.join( data_path, opt['gene2id'] )
	drug2id = os.path.join( data_path, opt['drug2id'] )
	cell2id = os.path.join( data_path, opt['cell2id'] )
	
	genotype = os.path.join(data_path, opt['genotype'])
	fingerprint= os.path.join(data_path, opt['fingerprint'])

	CUDA_ID = int(opt['cuda_id'])
	model_file = os.path.join(opt['output_dir'], 'model_best.pt') # gihan
	hidden_folder = opt['output_dir']
	batch_size=opt['batchsize']
	result_file=opt['output_dir']

	predict_data, cell2id_mapping, drug2id_mapping = prepare_predict_data(predict, cell2id, drug2id, opt['train_subset'])

	# getting drug and cell names
	infer_data=[]
	with open(predict, 'r') as fi:
		for il, line in enumerate(fi):
			tokens = line.strip().split('\t')
			cell_name = tokens[0]
			drug_name = tokens[1]
			cell_id = cell2id_mapping[cell_name]
			drug_id = drug2id_mapping[drug_name]

			infer_data.append([cell2id, cell_name, drug_id, drug_name])
	infer_data = pd.DataFrame(infer_data, columns=['cell2id', 'cell_name', 'drug_id', 'drug_name'])
	# getting drug and cell names

	gene2id_mapping = load_mapping(gene2id)

	# load cell/drug features
	cell_features = np.genfromtxt(genotype, delimiter=',')
	drug_features = np.genfromtxt(fingerprint, delimiter=',')

	num_cells = len(cell2id_mapping)
	num_drugs = len(drug2id_mapping)
	num_genes = len(gene2id_mapping)
	gene_dim=num_genes
	drug_dim = len(drug_features[0,:])

	feature_dim = gene_dim + drug_dim

	model = torch.load(model_file, map_location='cuda:%d' % CUDA_ID)

	predict_feature, predict_label = predict_data

	predict_label_gpu = predict_label.cuda(CUDA_ID)

	model.cuda(CUDA_ID)
	model.eval()

	test_loader = du.DataLoader(du.TensorDataset(predict_feature,predict_label), batch_size=batch_size, shuffle=False)

	#Test
	test_predict = torch.zeros(0,0).cuda(CUDA_ID)
	term_hidden_map = {}	

	batch_num = 0
	for i, (inputdata, labels) in enumerate(test_loader):
		# Convert torch tensor to Variable
		features = build_input_vector(inputdata, cell_features, drug_features)

		cuda_features = Variable(features.cuda(CUDA_ID), requires_grad=False)

		# make prediction for test data
		aux_out_map, term_hidden_map = model(cuda_features)

		if test_predict.size()[0] == 0:
			test_predict = aux_out_map['final'].data
		else:
			test_predict = torch.cat([test_predict, aux_out_map['final'].data], dim=0)

		# print("test_predict::: ", aux_out_map['final'].data)
		# suppressing the writing of these hidden files - gihan
		# for term, hidden_map in term_hidden_map.items():
		# 	hidden_file = hidden_folder+'/'+term+'.hidden'
		# 	with open(hidden_file, 'ab') as f:
		# 		np.savetxt(f, hidden_map.data.cpu().numpy(), '%.4e')
		# suppressing the writing of these hidden files - gihan

		batch_num += 1

	df_res = pd.DataFrame(zip(infer_data.loc[:, 'drug_name'] ,predict_label.detach().cpu().numpy().ravel().tolist(),
	 test_predict.detach().cpu().numpy().ravel().tolist()), columns=['smiles',  'true', 'pred'])
	test_corr = pearson_corr(test_predict, predict_label_gpu)
	test_corr = test_corr.cpu().numpy().item()
	test_predict_cpu = test_predict.cpu().numpy()
	predict_label_cpu = predict_label_gpu.cpu().numpy()

	test_spear = spearmanr(test_predict_cpu, predict_label_cpu)[0]
	

	# print("Test pearson corr\t%s\t%.6f" % (model.root, test_corr))
	np.savetxt(result_file+'/drugcell.predict', test_predict.cpu().numpy(),'%.4e')
	return {'test_corr':test_corr, 'spearmanr':test_spear}, df_res
