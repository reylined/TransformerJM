import pickle
import numpy as np
import pandas as pd


## Simulations

path = 'PATH'

auc_df = pd.DataFrame(np.nan, index=['none','interaction','nonph'],
                      columns=['true','MFPCA_Cox','MFPCA_DS','MATCH','Transformer'])
bs_df = pd.DataFrame(np.nan, index=['none','interaction','nonph'],
                     columns=['true','MFPCA_Cox','MFPCA_DS','MATCH','Transformer'])
rmse_df = pd.DataFrame(np.nan, index=['none','interaction','nonph'],
                       columns=['MFPCA_Cox','Transformer'])

models = ['MFPCA_Cox','MFPCA_DS','MATCH','Transformer']
scenario = ["none","interaction","nonph"]

landmark_times = [1,2,3,4,5]
pred_windows = [1,2,3]

def get_val(results, LTs, pred_win):
    def get_integrated(x, times):
        return np.trapz(x,times) / (max(times)-min(times))
    auc = results['AUC'][:,:,pred_win]
    bs = results['BS'][:,:,pred_win]
    true_auc = results['True_AUC'][:,:,pred_win]
    true_bs = results['True_BS'][:,:,pred_win]
    
    iauc = np.mean(get_integrated(auc, LTs))
    ibs = np.mean(get_integrated(bs, LTs))
    true_iauc = np.mean(get_integrated(true_auc, LTs))
    true_ibs = np.mean(get_integrated(true_bs, LTs))
    
    return iauc, ibs, true_iauc, true_ibs

def get_rmse(results):
    return np.mean(np.sqrt(np.mean(results['Long_MSE'], axis=1)))

for model in models:
    for case in scenario:
        file_name = model + '_' + case + '.pickle'
        infile = open(path + file_name, 'rb')
        results = pickle.load(infile)
        infile.close
        
        iauc, ibs, true_iauc, true_ibs = get_val(results, landmark_times, pred_win=0) # pred_win is index of pred_windows
        auc_df.loc[case,model] = iauc
        bs_df.loc[case,model] = ibs
        
        if model == "MFPCA_Cox":
            auc_df.loc[case,'true'] = true_iauc
            bs_df.loc[case,'true'] = true_ibs
            
            rmse_df.loc[case,model] = get_rmse(results)
            
        if model == "Transformer":
            rmse_df.loc[case,model] = get_rmse(results)

print(auc_df.round(3).to_latex())
print(bs_df.round(3).to_latex())
print(rmse_df.round(3).to_latex())










