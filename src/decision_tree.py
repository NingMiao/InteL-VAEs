import numpy as np
from sklearn.ensemble import RandomForestClassifier

def reorder(x_label):
    ind_by_label=[]
    for i in range(10):
        ind_by_label.append([])
    for i in range(len(x_label)):
        label=int(x_label[i])
        ind_by_label[label].append(i)
    ind_reorder=[]
    while len(ind_reorder)<len(x_label):
        for i in range(len(ind_by_label)):
            if len(ind_by_label[i])>0:
                ind_reorder.append(ind_by_label[i][0])
                del(ind_by_label[i][0])
    return ind_reorder

def decision_tree(x, y, train_size=10, test_size=2000, repeat_time=10):
    init_seed=10
    acc_list=[]
    for seed in range(repeat_time):
        np.random.seed(seed+init_seed)
        ind=np.arange(x.shape[0])
        np.random.shuffle(ind)
        x=x[ind]
        y=y[ind]
        ind_reorder=reorder(y)
        x=x[ind_reorder]
        y=y[ind_reorder]
        y=np.reshape(y, [-1])
        clf = RandomForestClassifier(max_depth=4, random_state=0, n_estimators=50)
        clf.fit(x[:train_size], y[:train_size]);
        acc = clf.score(x[-test_size:], y[-test_size:])
        acc_list.append(acc)
    return np.mean(acc_list)