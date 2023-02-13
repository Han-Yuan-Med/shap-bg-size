import numpy as np
import statistics

# As indicated in the paper, Jaccard index calculations are all percentile-based
def Jaccard(pred_seq, label_seq):  #@save
    """Compute the Jaccard."""
    # pred_tokens, label_tokens = pred_seq.split(","), label_seq.split(' ')
    len_pred = len(pred_seq)
    score = len_pred / len(np.unique(list(pred_seq)+list(label_seq)))
    return score


# 0% - 25% Jaccard calculation
Jaccard_25 = []
for b in range(99):
    for b2 in range((b+1), 100):
        m_tmp = Jaccard(orderlist[b][range(6)], orderlist[b2][range(6)])
        Jaccard_25.append([b, b2, m_tmp])

np.savetxt('Jaccard 50 25%.csv', Jaccard_25, delimiter=',')
# Jaccard_whole stores the Jaccard scores of all pairs
# Calculate the mean value of all pairs to get the stability of background sample size 50 using 100 simulations
print("Under 100 simulations, SHAP-based global ranking (0-25%) with background sample size of 50 evaluated "
      "by Jaccard is "+str(sum(elt[2] for elt in Jaccard_25)/len(Jaccard_25)))

# 25% - 50% Jaccard calculation
Jaccard_50 = []
for b in range(99):
    for b2 in range((b+1), 100):
        m_tmp = Jaccard(orderlist[b][range(6, 11)], orderlist[b2][range(6, 11)])
        Jaccard_50.append([b, b2, m_tmp])

np.savetxt('Jaccard 50 50%.csv', Jaccard_50, delimiter=',')
print("Under 100 simulations, SHAP-based global ranking (25-50%) with background sample size of 50 evaluated "
      "by Jaccard is "+str(sum(elt[2] for elt in Jaccard_50)/len(Jaccard_50)))

# 50% - 75% Jaccard calculation
Jaccard_75 = []
for b in range(99):
    for b2 in range((b+1), 100):
        m_tmp = Jaccard(orderlist[b][range(11, 16)], orderlist[b2][range(11, 16)])
        Jaccard_75.append([b, b2, m_tmp])

np.savetxt('Jaccard 50 75%.csv', Jaccard_75, delimiter=',')
print("Under 100 simulations, SHAP-based global ranking (50-75%) with background sample size of 50 evaluated "
      "by Jaccard is "+str(sum(elt[2] for elt in Jaccard_75)/len(Jaccard_75)))

# 75% - 100% Jaccard calculation
Jaccard_100 = []
for b in range(99):
    for b2 in range((b+1), 100):
        m_tmp = Jaccard(orderlist[b][range(16, 21)], orderlist[b2][range(16, 21)])
        Jaccard_100.append([b, b2, m_tmp])

np.savetxt('Jaccard 50 100%.csv', Jaccard_100, delimiter=',')
print("Under 100 simulations, SHAP-based global ranking (75-100%) with background sample size of 50 evaluated "
      "by Jaccard is "+str(sum(elt[2] for elt in Jaccard_100)/len(Jaccard_100)))
