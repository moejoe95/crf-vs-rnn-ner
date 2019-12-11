
def metrics_per_token(y_pred, y_act):
    tp = tn = fp = fn = 0
    for i, pred_label_sen in enumerate(y_pred):
        for j, pred_label in enumerate(pred_label_sen):

            if y_act[i][j] == '-PAD-' and pred_label == '-PAD-': 
                continue

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

    prec = rec = f1 = 0
    if tp + fp != 0:
        prec = tp / (tp + fp)
    if tp + fn != 0:
        rec = tp / (tp + fn)
    if prec + rec != 0:
        f1 = 2 * prec * rec/ (prec + rec)
    return prec, rec, f1

# see test-report for problems in this function
def metrics_per_ne(y_pred, y_act):
    tp = tn = fp = fn = 0
    for i, pred_label_sen in enumerate(y_pred):
        len_ne = 0
        for j, pred_label in enumerate(pred_label_sen):

            if y_act[i][j] == '-PAD-': 
                continue

            if y_act[i][j] != 'O' and y_act[i][j] != '-PAD-':
                k = j
                while(y_act[i][k] != 'O' and y_act[i][k]!= '-PAD-'):
                    k += 1
                    if k == len(pred_label_sen):
                        break
                len_ne = k - j
            else:
                len_ne = 0

            len_ne -= 1
            if len_ne <= 0:
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

    prec = rec = f1 = 0
    if tp + fp != 0:
        prec = tp / (tp + fp)
    if tp + fn != 0:
        rec = tp / (tp + fn)
    if prec + rec != 0:
        f1 = 2 * prec * rec/ (prec + rec)
    return prec, rec, f1


def print_conll_report(y_pred, y_act):
    print('\n')
    print('performance per token:')
    prec, rec, f1 = metrics_per_token(y_pred, y_act)
    print('\tprecision:\t', round(prec, 3))
    print('\trecall:\t\t', round(rec, 3))
    print('\tf1:\t\t', round(f1, 3))
    print('\n')
    print('performance per named entity:')
    prec, rec, f1 = metrics_per_ne(y_pred, y_act)
    print('\tprecision:\t', round(prec, 3))
    print('\trecall:\t\t', round(rec, 3))
    print('\tf1:\t\t', round(f1, 3))
