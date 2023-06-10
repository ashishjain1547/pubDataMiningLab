import math
import pandas as pd

def info_required_to_classify_a_tuple_in_data_D(df, label, rowid):
    total = df.shape[0]
    v = df.groupby([label])[rowid].count()
    l = []
    
    report = []
    
    for i in list(v.index):
        p = round(v[i]/total, 2)
        plogp = p * math.log2(p)
        report.append([i, v[i], p, plogp])
        
        l.append(plogp)
    
    df_report = pd.DataFrame(report, columns=['label', 'count', 'probability', 'p times logp'])
    return (df_report, -sum(l))
    
def info_required_after_split_on_attr(df_all, attr, label):

    l = []
    report_tuples = []
    
    for i in df_all[attr].unique():
        df = df_all[df_all[attr] == i]
        
        report, info = info_required_to_classify_a_tuple_in_data_D(df, 
                                                         label = label, 
                                                         rowid = 'RID')

        report_tuples.append([i, 
              round(df.shape[0] / df_all.shape[0], 2), 
              round(info, 2),
              round(df.shape[0] / df_all.shape[0], 2) * round(info, 2)
        ])
        l.append(round(df.shape[0] / df_all.shape[0], 2) * round(info, 2))
    
    df_report = pd.DataFrame(report_tuples, columns=['attr', 'data(attr == j) / data(all)', 'info', 'product of the two'])
    
    #print("Info still required if we split on age:", round(sum(l), 3))
    return (df_report, round(sum(l), 3))