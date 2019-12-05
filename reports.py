
def metrics(y_pred, y_act):
    tp = tn = fp = fn = 0
    for i, pred_label_sen in enumerate(y_pred):
        for j, pred_label in enumerate(pred_label_sen):
            if pred_label == y_act[i][j] and pred_label != 'O':
                tp += 1
            elif pred_label == y_act[i][j] and pred_label == 'O':
                tn += 1
            elif pred_label != y_act[i][j] and pred_label != 'O':
                fp += 1
            elif pred_label != y_act[i][j] and pred_label == 'O':
                fn += 1
            else:
                print('error: invalid data')
                return 0, 0, 0

    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * prec * rec/ (prec + rec)
    return prec, rec, f1