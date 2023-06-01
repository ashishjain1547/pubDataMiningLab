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
    