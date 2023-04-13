import datetime
import logging
from model import APICom
from utils import set_seed
import pandas as pd

set_seed(42)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

model_dict = {
    'codet5': r'E:\models\codet5-base',
    'plbart': r'E:\models\plbart-base',
    'unixcoder': r'E:\models\unixcoder-base',
    'codebert': r'E:\models\codebert-base'
}

model_type = 'codet5'

model = APICom(model_type=model_type, model_name_or_path=model_dict[model_type], load_model_path=None,
               beam_size=10, max_source_length=64, max_target_length=16)

start = datetime.datetime.now()

model.train(train_filename='../data/train.csv', train_batch_size=64, learning_rate=5e-5,
            num_train_epochs=50, early_stop=3, do_eval=True, eval_filename='../data/dev.csv',
            eval_batch_size=64, output_dir='valid_output/' + model_type + '/', do_eval_bleu=True)

end = datetime.datetime.now()
print(end - start)

for i in range(1, 6):
    model = APICom(model_type=model_type, model_name_or_path=model_dict[model_type], beam_size=10,
                   max_source_length=64, max_target_length=16,
                   load_model_path='valid_output/' + model_type + '/checkpoint-best-bleu/pytorch_model.bin',
                   num_return_sequences=i)

    model.test(batch_size=64, filename='../data/test.csv',
               output_dir=f'../results/beam_search_{i}')
