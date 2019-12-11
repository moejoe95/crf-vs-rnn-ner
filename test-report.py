import reports

act = [['O', 'I-PER', 'I-PER', 'I-PER', 'O']]
pred = [['O', 'I-PER', 'O', 'I-MISC', 'O']]

reports.print_conll_report(pred, act)


act = [['O', 'I-PER', 'I-PER', 'I-PER', 'O']]
pred = [['O', 'I-PER', 'O', 'I-PER', 'O']]

reports.print_conll_report(pred, act)
