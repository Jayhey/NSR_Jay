import argparse
import numpy as np
import multiprocessing as mp

from utils.keystroke import Family_R_A_KS_CM
from utils.measure import get_measure_df
from utils import Data_Handler, Hot_Fixer


def process(args):
    print('(Arguments: %s)' % args)
    try:
        Family_R_A_KS_CM(*args)
    except:
        print('뭔가가 잘못됨 흑흑 ...')
        raise Exception()


if __name__ == '__main__':
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--proc_n', type=int)
    parser.add_argument('--type', type=str)
    args = parser.parse_args()

    PROC_N = args.proc_n
    TYPE = args.type
    assert PROC_N and TYPE
    
    dh = Data_Handler()
    hf = Hot_Fixer(dh, TYPE)
    
    p = mp.Pool(PROC_N)
    arg_groups = [[TYPE, i, 10, 10, hf] for i in range(1, 11)]
    p.map(process, arg_groups)

    df = get_measure_df('./data/results', TYPE)
    df.to_csv('./data/reports/%s_EER_Results_Base.csv' % TYPE)
