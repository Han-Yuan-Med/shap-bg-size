import math
import collections
import numpy as np
import statistics


def bleu(pred_seq, label_seq, k):  #@save
    """Compute the BLEU."""
    # pred_tokens, label_tokens = pred_seq.split(","), label_seq.split(' ')
    len_pred, len_label = len(pred_seq), len(label_seq)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[''.join(str(label_seq[i:i + n]))] += 1
        for i in range(len_pred - n + 1):
            if label_subs[''.join(str(pred_seq[i:i + n]))] > 0:
                num_matches += 1
                label_subs[''.join(str(pred_seq[i:i + n]))] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


# Two grams BLEU calculation
# If users want to use other n grams, they can replace 2 with n: Bleu.bleu(orderlist[b], orderlist[b2], n)
# Whole sequence BLEU calculation
bleu_whole = []
for b in range(99):
    for b2 in range((b+1), 100):
        m_tmp = bleu(orderlist[b], orderlist[b2], 2)
        bleu_whole.append([b, b2, m_tmp])
np.savetxt('Bleu 50 whole.csv', bleu_whole, delimiter=',')
# bleu_whole stores the bleu scores of all pairs
# Calculate the mean value of all pairs to get the stability of background sample size 50 using 100 simulations
print("Under 100 simulations, SHAP-based global ranking with background sample size of 50 evaluated "
      "by BLEU is "+str(sum(elt[2] for elt in bleu_whole)/len(bleu_whole)))

# 25% sequence BLEU calculation
# If users prefer quartile-based BLUE, they can change Bleu.bleu(orderlist[b], orderlist[b2], 2)
# For example, features in the 25% quartile are of interest and there are 5 features in the 25% quartile
# m_tmp = Bleu.bleu(orderlist[b][range(6)], orderlist[b2][range(6)], 2)
bleu_25 = []
for b in range(99):
    for b2 in range((b+1), 100):
        m_tmp = bleu(orderlist[b][range(6)], orderlist[b2][range(6)], 2)
        bleu_25.append([b, b2, m_tmp])

np.savetxt('Bleu 50 25%.csv', bleu_25, delimiter=',')
print("Under 100 simulations, SHAP-based global ranking (0-25%) with background sample size of 50 evaluated "
      "by BLEU is "+str(sum(elt[2] for elt in bleu_25)/len(bleu_25)))

# 25% - 50% BLEU calculation
bleu_50 = []
for b in range(99):
    for b2 in range((b+1), 100):
        m_tmp = bleu(orderlist[b][range(6, 11)], orderlist[b2][range(6, 11)], 2)
        bleu_50.append([b, b2, m_tmp])

np.savetxt('Bleu 50 50%.csv', bleu_50, delimiter=',')
print("Under 100 simulations, SHAP-based global ranking (25-50%) with background sample size of 50 evaluated "
      "by BLEU is "+str(sum(elt[2] for elt in bleu_50)/len(bleu_50)))

# 50% - 75% BLEU calculation
bleu_75 = []
for b in range(99):
    for b2 in range((b+1), 100):
        m_tmp = bleu(orderlist[b][range(11, 16)], orderlist[b2][range(11, 16)], 2)
        bleu_75.append([b, b2, m_tmp])

np.savetxt('Bleu 50 50%.csv', bleu_75, delimiter=',')
print("Under 100 simulations, SHAP-based global ranking (50-75%) with background sample size of 50 evaluated "
      "by BLEU is "+str(sum(elt[2] for elt in bleu_75)/len(bleu_75)))

# 75% - 100% BLEU calculation
bleu_100 = []
for b in range(99):
    for b2 in range((b+1), 100):
        m_tmp = bleu(orderlist[b][range(16, 21)], orderlist[b2][range(16, 21)], 2)
        bleu_100.append([b, b2, m_tmp])

np.savetxt('Bleu 50 50%.csv', bleu_100, delimiter=',')
print("Under 100 simulations, SHAP-based global ranking (75-100%) with background sample size of 50 evaluated "
      "by BLEU is "+str(sum(elt[2] for elt in bleu_100)/len(bleu_100)))