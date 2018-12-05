import pickle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

fp = open('output.pkl','rb')
fo = open('actual.pkl','rb')
pred = pickle.load(fp)
y_test = pickle.load(fo)
y_pred = []

for p in pred:
    temp = [0 if float(x)<0.5 else 1 for x in p]
    y_pred.append(temp)

#print(len(y_test), len(pred))
f1 = 0
precision = 0
recall = 0

for i in range(len(y_pred)):
    f1+=f1_score(y_test[i], y_pred[i], average="macro")
    precision+=precision_score(y_test[i], y_pred[i], average="macro")
    recall+=recall_score(y_test[i], y_pred[i], average="macro")

print('F1 score:',f1/len(y_test))
print('precision:',precision/len(y_test))
print('recall:',recall/len(y_test))
