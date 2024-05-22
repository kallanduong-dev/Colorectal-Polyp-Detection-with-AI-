from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve
import matplotlib.pyplot as plt
import statistics
import numpy as np
import scipy
from scipy.stats import entropy
from sklearn.metrics import confusion_matrix

def avEntropy(arr):
    entropys = []
    
    
    for i in range(len(arr)):
        entropys.append(entropy(np.array([arr[i],1-arr[i]])))

    return [entropys, statistics.median(entropys)]


def entropyofAverage(arr):
    averageCon = statistics.median(arr)
    return entropy([averageCon,1-averageCon])


avcon = []

cases = []
for i in range(101):
    cases.append([])

bbox_areas = []
for i in range(100):
    bbox_areas.append([])

y_true = []
y_scores = []


missed_count = [0]*100

total_count = [0]*100



with open("entropy_results.txt") as filestream:
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

            #ignore inconsistent cases

            fileIndex = int(content[i].find("File"))+4
            if content[i][fileIndex+2].isnumeric(): 
                file = int(content[i][fileIndex:fileIndex+3])
            elif content[i][fileIndex+1].isnumeric():

                file = int(content[i][fileIndex:fileIndex+2])
            else:
                file = int(content[i][fileIndex])



            if case != 101:
                with open(f"annotation_txt/case{case}.txt","r") as newfilestream:
                    annotation = newfilestream.readlines()
                    line = annotation[file-1]
                    currentline = line.split(" ")
                    vals = currentline[1].split(",")
                    xmin = int(vals[0])
                    ymin = int(vals[1])
                    xmax = int(vals[2])
                    ymax = int(vals[3])
                    width = xmax - xmin
                    height = ymax - ymin
                    area = width*height

            maxcon = -1

            for j in range(i+1,i+10):
                if content[j][0] == 'E':
                    break
                if content[j][8] != '%':
                    confidence = float(content[j][7:9])/100
                else:
                    confidence = float(content[j][7])/100
                maxcon = max(confidence,maxcon)

            if maxcon != -1:
                cases[case-1].append(maxcon)
                if case != 101:
                    bbox_areas[case-1].append(area)
                    total_count[case-1] += 1
                y_scores.append(1)
            else:
                y_scores.append(0)
                if case != 101:
                    total_count[case-1] += 1
                    missed_count[case-1] += 1
                

            if case == 101:
                y_true.append(0)
            else:
                y_true.append(1)



for i in range(len(cases)):
    if len(cases[i]) > 0:
        if statistics.median(cases[i]) > 0:
            avcon.append(statistics.median(cases[i]))
        else:
            avcon.append(statistics.mean(cases[i]))
    else:
        avcon.append(0)


total_entropy = []
total_area = []

all_entropy = []
all_area = []
for i in range(len(cases)-1):
    for j in range(len(cases[i])):
            if bbox_areas[i][j] <= 200000:
                all_entropy.append(entropy(np.array([cases[i][j],1-cases[i][j]])))
                all_area.append(bbox_areas[i][j])
            total_entropy.append(entropy(np.array([cases[i][j],1-cases[i][j]])))
            total_area.append(bbox_areas[i][j])
            
            
plt.hist(total_area,bins=80)
plt.title('Number of Polyps in Each Bounding Box Size')
plt.xlabel('Bounding Box Area (Pixels)')
plt.ylabel('Count')
plt.show()

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(all_area, all_entropy)

print(r_value)

plt.scatter(all_area,all_entropy, alpha =0.2)
plt.title("Bounding Box Area vs Entropy")
plt.xlabel("Bounding Box Area (Pixels)")
plt.ylabel("Entropy")
plt.show()

all_entropy.sort()


filtered_cases = []
for i in range(101):
    filtered_cases.append([])

filtered_areas = []
for i in range(100):
    filtered_areas.append([])


for i in range(len(cases)-1):
    for j in range(len(cases[i])):
        if bbox_areas[i][j] > 50000 and bbox_areas[i][j] < 100000:
            filtered_cases[i].append(cases[i][j])
            filtered_areas[i].append(bbox_areas[i][j])

filtered_entropy = []


avcon.pop()


sizes = [3,18,6,4,3,3,6,12,4,3,5,5,5,3,5,4,4,2,3,3,3,2,12,15,7,5,5,2,13,4,12,15,5,3,15,7,7,5,13,5,3,7,10,5,3,2,5,3,3,10,15,6,4,4,3,4,5,6,8,8,6,7,7,3,3,6,5,15,3,15,4,5,3,5,3,3,4,12,4,10,6,3,13,5,8,4,3,4,5,10,13,7,7,6,8,5,15,4,5,3]

#5: IIA, 6: IS, 7: ISP, 8: IP

shape = [6,6,5,6,5,5,5,7,6,5,5,6,6,5,6,6,6,6,5,5,5,5,8,8,6,6,6,6,7,5,8,8,6,6,8,5,6,6,5,5,5,6,7,5,6,5,6,6,5,7,5,5,6,6,6,6,6,5,7,5,7,6,6,5,5,6,5,6,5,8,6,6,6,7,6,6,6,7,6,6,6,5,7,6,5,5,6,6,8,6,5,6,6,6,7,6,5,5,6,5]

#1. hyperplastic polyp, 2. sessile serated lesion, 3. low grade adenoma, 4. traditional serrated adenoma,  5. high grade adenoma, 6. invasive carcimona 

diagnosis = [3,5,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,1,3,3,3,3,3,3,3,3,1,3,3,3,3,4,3,3,5,3,3,3,3,3,3,3,3,3,3,1,3,3,3,3,3,3,1,3,3,1,3,2,4,3,3,3,6,3,3,3,3,5,3,3,3,3,3,3,3,3,3,5,3,3,3,2,3,3,3,3,3,3,3,2,3,3,3,3,3,1,2,3,3,1]

#1: cecum, 2: rectum, 3: Ascending Colon, 4: Sigmoid Colon, 5: Transverse Colon, 6: Descending Colon

location = [1,2,3,4,5,4,6,4,4,5,6,2,5,4,5,5,4,4,5,3,4,3,3,4,4,6,3,5,4,4,6,3,4,3,4,4,5,3,3,5,2,5,3,4,3,5,5,5,5,4,1,4,2,4,3,4,5,5,4,5,3,3,2,4,1,4,5,2,6,4,5,5,1,4,5,1,3,4,6,4,4,4,2,6,3,4,1,3,6,3,3,6,6,4,4,4,1,5,4,2]


#split shape into confidence 
IIA = []
IS = []
ISP = []
IP = []

for i in range(0,len(avcon)):
    if len(filtered_cases[i]) > 0:
        if shape[i] == 5:
            IIA.append(statistics.median(filtered_cases[i]))
        if shape[i] == 6:
            IS.append(statistics.median(filtered_cases[i]))
        if shape[i] == 7:
            ISP.append(statistics.median(filtered_cases[i]))
        if shape[i] == 8:
            IP.append(statistics.median(filtered_cases[i]))



IIA_ents = []

for i in range(len(IIA)):
    IIA_ents.append(entropy(np.array([IIA[i],1-IIA[i]])))

IS_ents = []

for i in range(len(IS)):
    IS_ents.append(entropy(np.array([IS[i],1-IS[i]])))

ISP_ents = []

for i in range(len(ISP)):
    ISP_ents.append(entropy(np.array([ISP[i],1-ISP[i]])))

IP_ents = []

for i in range(len(IP)):
    IP_ents.append(entropy(np.array([IP[i],1-IP[i]])))






#get average entropy for each shape

IIAAvEntropy = entropyofAverage(IIA)
ISAvEntropy = entropyofAverage(IS)
ISPAventropy = entropyofAverage(ISP)
IPAventropy = entropyofAverage(IP)

#plot graph for shape

entropys = [IIAAvEntropy,ISAvEntropy,ISPAventropy, IPAventropy]
# Numbers of pairs of bars you want
N = 4
# Specify the values of blue bars (height)

blue_bar = (statistics.median(IIA),statistics.median(IS),statistics.median(ISP),statistics.median(IP))
# Specify the values of orange bars (height)
orange_bar = entropys

# Position of bars on x-axis
ind = np.arange(N)

# Figure size
plt.figure(figsize=(10,5))

# Width of a bar 
width = 0.3       

# Plotting
plt.bar(ind, blue_bar , width, label='Average Confidence')
plt.bar(ind + width, orange_bar, width, label='Average Entropy')

plt.xlabel('Polyp Shape')
plt.ylabel('Average Confidence/Entropy')
plt.title('Average Confidence and Entropy of different Shaped Polyps')

# xticks()
# First argument - A list of positions at which ticks should be placed
# Second argument -  A list of labels to place at the given locations
plt.xticks(ind + width / 2, ('IIA', 'IS', 'ISP','IP'))

# Finding the best position for legends and putting it
plt.legend(loc='best')
plt.show()


#sort into location categories

C = []
R = []
AC = []
SC = []
TC = []
DC = []

for i in range(len(shape)):
    if filtered_cases[i]:
        if location[i] == 1:
            C.append(statistics.median(filtered_cases[i]))
        if location[i] == 2:
            R.append(statistics.median(filtered_cases[i]))
        if location[i] == 3:
            AC.append(statistics.median(filtered_cases[i]))
        if location[i] == 4:
            SC.append(statistics.median(filtered_cases[i]))
        if location[i] == 5:
            TC.append(statistics.median(filtered_cases[i]))
        if location[i] == 6:
            DC.append(statistics.median(filtered_cases[i]))

C_ents = []

for i in range(len(C)):
    C_ents.append(entropy(np.array([C[i],1-C[i]])))

R_ents = []

for i in range(len(R)):
    R_ents.append(entropy(np.array([R[i],1-R[i]])))

AC_ents = []

for i in range(len(AC)):
    AC_ents.append(entropy(np.array([AC[i],1-AC[i]])))

SC_ents = []

for i in range(len(SC)):
    SC_ents.append(entropy(np.array([SC[i],1-SC[i]])))

TC_ents = []

for i in range(len(TC)):
    TC_ents.append(entropy(np.array([TC[i],1-TC[i]])))

DC_ents = []

for i in range(len(DC)):
    DC_ents.append(entropy(np.array([DC[i],1-DC[i]])))



#get average entropy

CAvEntropy = entropyofAverage(C)
RAvEntropy = entropyofAverage(R)
ACAvEntropy = entropyofAverage(AC)
SCAvEntropy = entropyofAverage(SC)
TCAvEntropy = entropyofAverage(TC)
DCAvEntropy = entropyofAverage(DC)


#plot location graph

entropys = [CAvEntropy,RAvEntropy,ACAvEntropy,SCAvEntropy,TCAvEntropy,DCAvEntropy]
# Numbers of pairs of bars you want
N = 6

# Specify the values of blue bars (height)

blue_bar = (statistics.median(C),statistics.median(R),statistics.median(AC),statistics.median(SC),statistics.median(TC),statistics.median(DC))
# Specify the values of orange bars (height)
orange_bar = entropys

# Position of bars on x-axis
ind = np.arange(N)

# Figure size
plt.figure(figsize=(10,5))

# Width of a bar 
width = 0.3       

# Plotting
plt.bar(ind, blue_bar , width, label='Average Confidence')
plt.bar(ind + width, orange_bar, width, label='Average Entropy')

plt.xlabel('Polyp Location')
plt.ylabel('Average Confidence/Entropy')
plt.title('Average Confidence and Entropy of Polyps From Different Locations')

# xticks()
# First argument - A list of positions at which ticks should be placed
# Second argument -  A list of labels to place at the given locations
plt.xticks(ind + width / 2, ('Cecum', 'Rectum','Ascending Colon','Sigmoid Colon','Transverse Colon','Descending Colon'))

# Finding the best position for legends and putting it
plt.legend(loc='best')
plt.show()

