
def get_validation(y_pred, test_labels):
    tp = tn = fp = fn = 0

    for i in range(0, len(y_pred)-1):
        for j in range(0, len(y_pred[i])-1):
            p = y_pred[i][j]
            t = test_labels[i][j]

            if p != 'O' and t != 'O':
                tp += 1
            elif p == 'O' and t == 'O':
                tn +=1
            elif p != 'O' and t == 'O':
                fp += 1
            elif p == 'O' and t != 'O':
                fn += 1 
            else:
                print('test data invalid!')

    accuracy = (tp+tn)/(tp+tn+fp+fn) 
    precision = (tp) / (tp+fp) 
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return accuracy, precision, recall, f1
