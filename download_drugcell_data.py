import os
import candle
import urllib

CANDLE_DATA_DIR=os.getenv("CANDLE_DATA_DIR")
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
     },
    {'name': 'download_data',
     'type': bool,
     'help': '.....'
     }
]

required = None

file_path = os.path.dirname(os.path.realpath(__file__))


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
    download_ccle_data(opt)


    print("Done.")
