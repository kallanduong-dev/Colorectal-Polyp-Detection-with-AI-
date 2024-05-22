from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve
import matplotlib.pyplot as plt
import statistics
import numpy as np
from scipy.stats import entropy
from sklearn.metrics import confusion_matrix


avcon = []

cases = []
for i in range(101):
    cases.append([])

y_true = []
y_scores = []

with open("performance_results.txt") as filestream:
    content = filestream.readlines()
    for i in range(len(content)):
        if content[i].find("case") >= 0:
            caseIndex = int(content[i].find("case"))+4
            if content[i][caseIndex+2].isnumeric(): 
                case = int(content[i][caseIndex:caseIndex+3])
            elif content[i][caseIndex+1].isnumeric():
                case = int(content[i][caseIndex:caseIndex+2])
            else:
                case = int(content[i][caseIndex])

            fileIndex = int(content[i].find("File"))+4
            if content[i][fileIndex+2].isnumeric(): 
                file = int(content[i][fileIndex:fileIndex+3])
            elif content[i][fileIndex+1].isnumeric():

                file = int(content[i][fileIndex:fileIndex+2])
            else:
                file = int(content[i][fileIndex])

            if i+2 < len(content) and content[i+2].find("polyp: ") >= 0 or content[i+1].find("polyp: ") >= 0:
                confidence = float(content[i+1][7:9])/100
                cases[case-1].append(confidence)
                y_scores.append(1)
            else:
                y_scores.append(0)


            if case == 101:
                y_true.append(0)
            else:
                y_true.append(1)



fpr, tpr, thresholds = roc_curve(y_true, y_scores)

plt.plot(fpr, tpr)
plt.ylabel("True Positive Rate (TPR)")
plt.xlabel("False Positive Rate (FPR)")
plt.savefig('ROCCurve.png')
plt.show()

tn, fp, fn, tp = confusion_matrix(y_true, y_scores).ravel()

precision = tp/(tp+fn)
recall = tp/(tp+fp)

f1 = 2*((precision*recall)/(precision+recall))

print(precision, recall, f1)