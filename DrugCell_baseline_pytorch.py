import sys
import os
import numpy as np
import torch
import torch.utils.data as du
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import util
from util import *
from drugcell_NN import *
import argparse
import numpy as np
import time
import candle
import json
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from preprocess import preprocess_ccle

file_path = os.path.dirname(os.path.realpath(__file__))
additional_definitions = [
    {'name': 'config_file',
     'type': str,
     'help': '...'
     },
     {'name': 'epochs',
     'type': int,
     'help': '...'
     },
    {'name': 'data_path',
     'type': str,
     'help': '...'
     },
    {'name': 'onto',
     'type': str,
     'help': '...'
     },
    {'name': 'train',
     'type': str,
     'help': '...'
     },
    {'name': 'test',
     'type': str,
     'help': '...'
     },
    {'name': 'infer',
     'type': str,
     'help': '.....'
     },
    {'name': 'lr',
     'type': float,
     'help': '....'
     },
    {'name': 'batchsize',
     'type': int,
     'help': '..'
     },
    {'name': 'cuda',
     'type': int,
     'help': '.....'
     },
    {'name': 'gene2id',
     'type': str,
     'help': '...'
     },
    {'name': 'drug2id',
     'type': str,
     'help': '...'
     },
    {'name': 'cell2id',
     'type': str,
     'help': '...'
     },
    {'name': 'genotype_hiddens',
     'type': int,
     'help': '...'
     },
    {'name': 'drug_hiddens',
     'type': str,
     'help': '...'
     },
    {'name': 'final_hiddens',
     'type': int,
     'help': '...'
     },
    {'name': 'genotype',
     'type': str,
     'help': '...'
     },
    {'name': 'fingerprint',
     'type': str,
     'help': '.....'
     },
    {'name': 'train_subset',
     'type': int,
     'help': '.....'
     },
    {'name': 'data_split_seed',
     'type': int,
     'help': '.....'
     }
]

required = None


if os.getenv("CUDA_VISIBLE_DEVICES") is not None:
    print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))
    cuda_name = os.getenv("CUDA_VISIBLE_DEVICES")
else:
    cuda_name = 0

CUDA_ID = int(cuda_name)
CANDLE_DATA_DIR=os.getenv("CANDLE_DATA_DIR")


# build mask: matrix (nrows = number of relevant gene set, ncols = number all genes)
# elements of matrix are 1 if the corresponding gene is one of the relevant genes
def create_term_mask(term_direct_gene_map, gene_dim):

        term_mask_map = {}

        for term, gene_set in term_direct_gene_map.items():

                mask = torch.zeros(len(gene_set), gene_dim)

                for i, gene_id in enumerate(gene_set):
                        mask[i, gene_id] = 1

                mask_gpu = torch.autograd.Variable(mask.cuda(CUDA_ID))

                term_mask_map[term] = mask_gpu

        return term_mask_map

def eval_model(model, test_loader, cell_features, drug_features, CUDA_ID):
        model.eval()

        test_predict = torch.zeros(0,0).cuda(CUDA_ID)

        for i, (inputdata, labels) in enumerate(test_loader):
                # Convert torch tensor to Variable
                features = build_input_vector(inputdata, cell_features, drug_features)
                cuda_features = Variable(features.cuda(CUDA_ID))

                aux_out_map, _ = model(cuda_features)

                if test_predict.size()[0] == 0:
                        test_predict = aux_out_map['final'].data
                else:
                        test_predict = torch.cat([test_predict, aux_out_map['final'].data], dim=0)
        return test_predict


def train_model(root, term_size_map, term_direct_gene_map, dG, train_data, gene_dim, drug_dim, model_save_folder, train_epochs, batch_size,\
 learning_rate, num_hiddens_genotype, num_hiddens_drug, num_hiddens_final, cell_features, drug_features):
    


        epoch_start_time = time.time()
        best_model = 0
        max_corr = 0
        best_mse = 1e8

        # dcell neural network
        model = drugcell_nn(term_size_map, term_direct_gene_map, dG, gene_dim, drug_dim, root, num_hiddens_genotype, num_hiddens_drug, num_hiddens_final)

        train_feature, train_label, test_feature, test_label = train_data

        train_label_gpu = torch.autograd.Variable(train_label.cuda(CUDA_ID))
        test_label_gpu = torch.autograd.Variable(test_label.cuda(CUDA_ID))

        model.cuda(CUDA_ID)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=1e-05)
        term_mask_map = create_term_mask(model.term_direct_gene_map, gene_dim)

        optimizer.zero_grad()

        for name, param in model.named_parameters():
                term_name = name.split('_')[0]

                if '_direct_gene_layer.weight' in name:
                        param.data = torch.mul(param.data, term_mask_map[term_name]) * 0.1
                else:
                        param.data = param.data * 0.1

        train_loader = du.DataLoader(du.TensorDataset(train_feature,train_label), batch_size=batch_size, shuffle=False)
        test_loader = du.DataLoader(du.TensorDataset(test_feature,test_label), batch_size=batch_size, shuffle=False)

        for epoch in range(train_epochs):

                #Train
                model.train()
                train_predict = torch.zeros(0,0).cuda(CUDA_ID)

                for i, (inputdata, labels) in enumerate(train_loader):
                        # Convert torch tensor to Variable
                        features = build_input_vector(inputdata, cell_features, drug_features)

                        cuda_features = torch.autograd.Variable(features.cuda(CUDA_ID))
                        cuda_labels = torch.autograd.Variable(labels.cuda(CUDA_ID))

                        # Forward + Backward + Optimize
                        optimizer.zero_grad()  # zero the gradient buffer

                        # Here term_NN_out_map is a dictionary 
                        aux_out_map, _ = model(cuda_features)

                        if train_predict.size()[0] == 0:
                                train_predict = aux_out_map['final'].data
                        else:
                                train_predict = torch.cat([train_predict, aux_out_map['final'].data], dim=0)

                        total_loss = 0  
                        for name, output in aux_out_map.items():
                                loss = nn.MSELoss()
                                if name == 'final':
                                        total_loss += loss(output, cuda_labels)
                                else: # change 0.2 to smaller one for big terms
                                        total_loss += 0.2 * loss(output, cuda_labels)

                        total_loss.backward()

                        for name, param in model.named_parameters():
                                if '_direct_gene_layer.weight' not in name:
                                        continue
                                term_name = name.split('_')[0]
                                #print name, param.grad.data.size(), term_mask_map[term_name].size()
                                param.grad.data = torch.mul(param.grad.data, term_mask_map[term_name])

                        optimizer.step()

                train_corr = pearson_corr(train_predict, train_label_gpu)
                #if epoch % 10 == 0:model_save_folder = os.path.join(CANDLE_DATA_DIR,model_save_folder)
                model_save_folder = os.path.join(CANDLE_DATA_DIR, model_save_folder)  # gihan
                os.makedirs(model_save_folder, exist_ok=True)  # gihan
                # torch.save(model, model_save_folder + '/model_' + str(epoch) + '.pt')

                #Test: random variables in training mode become static
                model.eval()
                
                test_predict = torch.zeros(0,0).cuda(CUDA_ID)

                for i, (inputdata, labels) in enumerate(test_loader):
                        # Convert torch tensor to Variable
                        features = build_input_vector(inputdata, cell_features, drug_features)
                        cuda_features = Variable(features.cuda(CUDA_ID))

                        aux_out_map, _ = model(cuda_features)

                        if test_predict.size()[0] == 0:
                                test_predict = aux_out_map['final'].data
                        else:
                                test_predict = torch.cat([test_predict, aux_out_map['final'].data], dim=0)


                        test_corr = pearson_corr(test_predict, test_label_gpu)
                        test_mse = mean_squared_error(y_pred=test_predict.cpu(), y_true = test_label_gpu.cpu())

                epoch_end_time = time.time()
                print("epoch\t%d\tcuda_id\t%d\ttrain_corr\t%.6f\tval_corr\t%.6f\ttotal_loss\t%.6f\telapsed_time\t%s" % (epoch, CUDA_ID, train_corr, test_corr, \
                total_loss, epoch_end_time-epoch_start_time))
                epoch_start_time = epoch_end_time

                # if test_corr >= max_corr:
                #         max_corr = test_corr
                #         best_model = epoch
                #         torch.save(model, model_save_folder + '/model_best' + '.pt')

                if test_mse <= best_mse:
                        best_mse = test_mse
                        best_model = epoch
                        torch.save(model, model_save_folder + '/model_best' + '.pt')


        # torch.save(model, model_save_folder + '/model_final.pt')        

        print("Best performed model (epoch)\t%d" % best_model)


        model_best = torch.load(model_save_folder + '/model_best' + '.pt') # gihan
        # model_best = torch.load(model_save_folder + '/model_' + str(best_model) + '.pt') # gihan
        # torch.save(model_best, model_save_folder + '/model_best.pt') # gihan

        test_predicted = eval_model(model_best, test_loader, cell_features, drug_features, CUDA_ID)
        test_predicted_cpu = test_predicted.cpu().numpy()
        test_label_cpu = test_label_gpu.cpu().numpy()

        test_corr = pearson_corr(test_predicted, test_label_gpu)
        test_spear = spearmanr(test_predicted_cpu, test_label_cpu)[0]
        mse = mean_squared_error(test_predicted_cpu, test_label_cpu)

        # print('labels')
        # print(test_predicted_cpu)
        # print(test_label_cpu)
        
        print('MSE: ', mse)
        print('test sp:', test_spear)
        # print('test_predicted:', test_predicted)
        # print('test_label_gpu:', test_label_gpu)
        test_scores = {"val_loss": str(mse),
         "pcc": str(test_corr.cpu().numpy().item()),
         "spearmanr": str(test_spear) }

        # gihan
        # with open( os.path.join(model_save_folder,"test_scores.json"), "w", encoding="utf-8") as f:
        #         json.dump(test_scores, f, ensure_ascii=False, indent=4)

        return test_scores


def get_data(data_url, cache_subdir, download=True):
    
    # cache_subdir = os.path.join(CANDLE_DATA_DIR, 'SWnet', 'Data')
    
    if download:
        print('downloading data')
        os.makedirs(cache_subdir, exist_ok=True)
        os.system(f'svn checkout {data_url} {cache_subdir}')   
        print('downloading done') 
    else:
        print('not downloading. data already present')


# def new_split_train_test(opt, data_path):
#     train = os.path.join(data_path, opt['train'])  
#     test = os.path.join(data_path, opt['test']) # for validation
#     infer = os.path.join(data_path, opt['infer'])

#     train_df = pd.read_csv(train, header=None, delimiter='\t')
#     val_df = pd.read_csv(test, header=None, delimiter='\t')
#     test_df = pd.read_csv(infer, header=None, delimiter='\t')
#     df = pd.concat([train_df,val_df,test_df], axis=0)
#     df.reset_index(drop=True, inplace=True)

#     train, test = train_test_split(df, test_size=0.2, random_state=opt['data_split_seed'])
#     val, test = train_test_split(test, test_size=0.5, random_state=opt['data_split_seed'])

#     train.to_csv(opt['output_dir']+'/train.txt', sep='\t', index=False, header=None)
#     val.to_csv(opt['output_dir']+'/val.txt', sep='\t', index=False, header=None)
#     test.to_csv(opt['output_dir']+'/test.txt', sep='\t', index=False, header=None)



def run(opt):
#     data_path=opt['data_path']

    base_path=os.path.join(CANDLE_DATA_DIR, opt['model_name'], 'Data')
    data_path = base_path

    if opt['data_type']=='ccle_candle':
        preprocess_ccle(opt)
    else:
        data_url = opt['data_url']
        download_data = opt['download_data']
        get_data(data_url, base_path, download_data)

    
    onto = os.path.join(data_path, opt['onto'])
    train = os.path.join(data_path, opt['train'])  
    test = os.path.join(data_path, opt['test']) # for validation
    infer = os.path.join(data_path, opt['infer'])
#     new_split_train_test(opt, data_path)
#     train = os.path.join(opt['output_dir']+'/train.txt')  
#     test = os.path.join(opt['output_dir']+'/val.txt') # for validation
#     infer = os.path.join(opt['output_dir']+'/test.txt')


    epoch = int(opt['epochs'])
    lr = float(opt['lr'])
    batchsize = int(opt['batchsize'])
    modeldir = opt['output_dir']
    cuda = int(opt['cuda'])

    gene2id = os.path.join(data_path, opt['gene2id'] )
    drug2id=  os.path.join(data_path, opt['drug2id'] )
    cell2id = os.path.join(data_path, opt['cell2id'] )

    genotype_hiddens = int(opt['genotype_hiddens'])
    drug_hiddens = opt['drug_hiddens']
    final_hiddens = int(opt['final_hiddens'])

    genotype= os.path.join(data_path, opt['genotype'])
    fingerprint= os.path.join(data_path, opt['fingerprint'])
    train_subset = opt['train_subset']


    # call functions
    torch.set_printoptions(precision=5)

    # load input data
    train_data, cell2id_mapping, drug2id_mapping = prepare_train_data(train, test, cell2id, drug2id, train_subset)
    gene2id_mapping = load_mapping(gene2id)

    # load cell/drug features
    cell_features = np.genfromtxt(genotype, delimiter=',')
    drug_features = np.genfromtxt(fingerprint, delimiter=',')

    num_cells = len(cell2id_mapping)
    num_drugs = len(drug2id_mapping)
    num_genes = len(gene2id_mapping)
    drug_dim = len(drug_features[0,:])

    # load ontology
    dG, root, term_size_map, term_direct_gene_map = load_ontology(onto, gene2id_mapping)

    # load the number of hiddens #######
    num_hiddens_genotype = genotype_hiddens

    num_hiddens_drug = list(map(int, drug_hiddens.split(',')))


    num_hiddens_final = final_hiddens
    #####################################

    # CUDA_ID = opt.cuda  # gihan

    test_scores = train_model(root, term_size_map, term_direct_gene_map, dG, train_data, num_genes, drug_dim, modeldir, epoch, \
    batchsize, lr, num_hiddens_genotype, num_hiddens_drug, num_hiddens_final, cell_features, drug_features)

    print("TEST SCORES")
    print(test_scores)
    with open(opt['output_dir'] + "/scores.json", "w", encoding="utf-8") as f:
        json.dump(test_scores, f, ensure_ascii=False, indent=4)
    print('IMPROVE_RESULT RMSE val_loss:\t' + str(test_scores['pcc'] ))

    infer_scores, df_infer = predict_dcell(opt, data_path, infer)
    infer_data = pd.read_csv(os.path.join(data_path,'drugcell_test.txt'), header=None, sep='\t')
    infer_data.columns = ['cell_line_id','drug_id', 'labels']
    df_infer = pd.concat([df_infer, infer_data], axis=1)
#     with open(opt['output_dir'] + "/scores_infer.json", "w", encoding="utf-8") as f:
#         json.dump(infer_scores, f, ensure_ascii=False, indent=4)
    print('IMPROVE_RESULT RMSE (INFER):\t', infer_scores)
    print(f"Writing the predictions to {opt['output_dir']} test_predictions.csv")
    df_infer.to_csv(opt['output_dir'] + "/test_predictions.csv", index=False)
    print("Predictions saved")


    return test_scores, infer_scores


class DrugCell_candle(candle.Benchmark):

        def set_locals(self):
            if required is not None:
                self.required = set(required)
            if additional_definitions is not None:
                self.additional_definitions = additional_definitions


def initialize_parameters():
    """ Initialize the parameters for the GraphDRP benchmark. """
    print("Initializing parameters\n")
    drugcell_params = DrugCell_candle(
                            filepath=file_path,
                            defmodel="drugcell_model.txt",
                                            # defmodel="graphdrp_model_candle.txt",
                            framework="pytorch",
                            prog="DrugCell",
                            desc="CANDLE compliant DrugCell",
                                )
    gParameters = candle.finalize_parameters(drugcell_params)
    return gParameters

if __name__ == '__main__':

    opt = initialize_parameters()
    print(opt)
    val_scores, infer_scores = run(opt)


    print("Done.")
