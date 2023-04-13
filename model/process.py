import pandas as pd
import numpy as np
from tqdm import *


df=pd.read_csv(r'E:\yangshaoyu\api_completion\results\beam_search_1\test_token.csv')

df=df['answer']
for i in range(1,11):
    df.to_csv(fr'E:\yangshaoyu\api_completion\results\beam_search_{i}\gold_token.csv',index=False,header=None)