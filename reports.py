
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
        j = 0
        while j < len(pred_label_sen):
            pred_label = y_pred[i][j]

            if y_act[i][j] == '-PAD-': 
                j += 1
                continue

            if y_act[i][j] != 'O':
                k = j
                while y_act[i][j] == y_act[i][k]:
                    k += 1
                    if k == len(pred_label_sen) or y_act[i][k] == '-PAD-':
                        break

                isExact = True
                for l in range(k - j):
                    if y_pred[i][j+l] != y_act[i][j+l]:
                        isExact = False
                if isExact:
                    tp += 1
                else:
                    fp += 1
                j = k - 1
            else:
                if pred_label == y_act[i][j]:
                    tn += 1
                elif pred_label != y_act[i][j]:
                    fn += 1
            j += 1

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
