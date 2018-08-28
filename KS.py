import os
from utils import Data_Handler
import multiprocessing
from Jay_module import *
from tqdm import tqdm
import time

# dh = Data_Handler()
# dh.save('./dh')
dh = Data_Handler('./dh')
output_path = './data/results'

training_amount = 10
test_amount = 10
num_iter = 10

sig = [0, 18, 24] # 별 OX 폰
user = np.array(['1','2','3','4'])

colname = ['KS_only_axis', 'KS_only_gyro', 'KS_both_axis_gyro']
index = [[0,1],[2,3,4],[0,1,2,3,4]]

# dh.datasets4sig['sig-st']['사용자번호(1~4)'][도형 번호(30개)][몇 번째?(30번)]

# 처음 5개, 마지막 5개 제거
for i in range(len(dh.datasets4sig['sig-st'])):
    for j in range(len(dh.datasets4sig['sig-st']['1'])):
        del dh.datasets4sig['sig-st'][str(i+1)][j][0:5]
        del dh.datasets4sig['sig-st'][str(i+1)][j][-5:]


# In[3]:
def main(fig):
    # sig_df = []
    iter_df = []
    for user_ in tqdm(user, leave=False):
        for i in tqdm(range(num_iter), leave=False):
            idx = np.array(range(20))
            r = np.random.choice(20, training_amount, replace=False)
            tr_idx = idx[r]
            te_idx = np.delete(idx, r)
            te_idx = np.random.choice(te_idx, test_amount, replace=False)

            valid_tr_ = np.array(dh.datasets4sig['sig-st'][user_][fig])[tr_idx]
            valid_te = np.array(dh.datasets4sig['sig-st'][user_][fig])[te_idx]
            test_te = np.append([np.array(dh.datasets4sig['sig-st'][user[np.in1d(user, user_,invert=True)][0]][fig])[te_idx], 
                                np.array(dh.datasets4sig['sig-st'][user[np.in1d(user, user_,invert=True)][1]][fig])[te_idx]],
                                np.array(dh.datasets4sig['sig-st'][user[np.in1d(user, user_,invert=True)][2]][fig])[te_idx])
            # Valid TR 데이터 획별로 하나로 합치기
            table = valid_tr_[0]['table']
            stroke_idx = np.where((table['Up-Down'].values== 'DOWN') | (table['Up-Down'].values=='UP'))[0]
            stroke_num = int(len(stroke_idx)/2)

            valid_tr = {str(i+1):[] for i in range(stroke_num)}
            for log in valid_tr_:
                tmp = diff_dataframe(log['table'])
                for k,v in tmp.items():
                    valid_tr[k].append(v.diff().dropna())

            valid_tr = {k:pd.concat(valid_tr[k]) for k in valid_tr.keys()}

            # Test & Validation
            test_df, valid_df = [], []
            for log in test_te:
                test_df.append(KS_calculator(valid_tr, log['table']))
            test_df = pd.concat(test_df)    

            for log in valid_te:
                valid_df.append(KS_calculator(valid_tr, log['table']))
            valid_df = pd.concat(valid_df)


            def final_df(valid_df, test_df):

                def transform_df(df, valid=True):
                    df =pd.concat([df.iloc[:,idx].mean(axis=1) for idx in index], axis=1)
                    df = df.reset_index(drop=True)
                    df.columns = colname

                    if valid:
                        target_idx = pd.DataFrame({"Target":np.ones(len(df)), "idx":tr_idx})
                        return pd.concat([df, target_idx], axis=1)
                    else:
                        target_idx = pd.DataFrame({"Target":np.zeros(len(df)), 
                                                "idx":np.tile(te_idx, len(user)-1)})
                        return pd.concat([df, target_idx], axis=1)

                final_test_df = transform_df(test_df, valid=False)
                final_valid_df = transform_df(valid_df)

                df = pd.concat([final_valid_df, final_test_df], axis=0)
                return df

            ks_df = final_df(valid_df, test_df)
            tmp_result = get_df_(ks_df)

            iter_df.append(tmp_result)


    result_df = pd.concat(iter_df)
    mean_std_df = pd.DataFrame([result_df.mean(), result_df.std()], index=["Mean","Std"])
    mean_std_df.to_csv(os.path.join(output_path, "{}_results.csv".format(fig)))

#    final_result = pd.concat(sig_df)
#    final_result.to_csv(os.path.join(output_path, "fig_pilot_results.csv"))


if __name__ == '__main__':    
    start_time = time.time()
    pool = multiprocessing.Pool(processes=4)
    pool.map(main, sig)
    print("--- %s seconds ---" % (time.time() - start_time))
    
