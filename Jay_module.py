import numpy as np
import pandas as pd


def diff_dataframe(table):
    stroke_idx = np.where((table['Up-Down'].values== 'DOWN') | (table['Up-Down'].values=='UP'))[0]
    stroke_num = int(len(stroke_idx)/2)
    stroke_dict = {}
    for i in range(stroke_num):
        tmp = table.iloc[stroke_idx[i*2:i*2+2][0]:stroke_idx[i*2:i*2+2][1]]
        stroke_dict[str(i+1)] = tmp.iloc[:,3:].diff().dropna()
    return stroke_dict
    

def KS_calculator(a,b, unit=0.1):
    # a = diff_dataframe(a, train=True)
    b = diff_dataframe(b)
    
    KS_dict = {}
    def KS_function(a, b, unit=unit):
        X_axis = np.arange(np.min(np.append(a, b)), np.max(np.append(a, b)), unit)
        tmp = np.array([])
        
        for X in X_axis:
            A_Y_Axis = len(np.where(a <= X)[0]) / len(a)
            B_Y_Axis = len(np.where(b <= X)[0]) / len(b)
            tmp = np.append(tmp, np.absolute(A_Y_Axis - B_Y_Axis))
        KS = np.max(tmp) 
        return KS

    for col in a['1'].columns:
        KS_dict[col] = np.mean([KS_function(a[str(i+1)][col], b[str(i+1)][col]) for i in range(len(a))])

    df = pd.DataFrame(data=KS_dict, index=[0])
    return df

def get_df_(ks_df):
    se_target = ks_df['Target']
    columns = ks_df.columns[:-2]
    results = []
    for col in columns:
        se = ks_df[col]
        cut_offs = sorted(set(se), reverse=True)
        DIFF, FAR, FRR = [], [], []
        for cut_off in cut_offs[:-1]:
            conds = (se >= cut_off)
            Predict_All_True_Upper = se_target[conds]
            False_Acceptance_Rate = (sum(Predict_All_True_Upper) / len(Predict_All_True_Upper))
            Predict_All_False_Under = se_target[~conds]
            False_Rejection_Rate = (sum(Predict_All_False_Under == 0) / len(Predict_All_False_Under))
            DIFF.append(abs(False_Acceptance_Rate-False_Rejection_Rate))
            FAR.append(False_Acceptance_Rate)
            FRR.append(False_Rejection_Rate)
            #print(col, cut_off)
            #print(Predict_All_False_Under, Predict_All_True_Upper)
        DIFF = np.array(DIFF)
        FAR8FRR = np.array([FAR, FRR]).T
        result = FAR8FRR[DIFF == DIFF.min()].sum()/2
        results.append(result)
    tmp_df = pd.DataFrame([results], columns=columns)
    return tmp_df
