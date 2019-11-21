# Results

## CRF classification report

### CoNLL 2003 - testa dataset
```
              precision    recall  f1-score   support

      B-MISC       0.00      0.00      0.00         4
       I-LOC       0.76      0.48      0.59      2094
      I-MISC       0.75      0.58      0.66      1264
       I-ORG       0.57      0.57      0.57      2092
       I-PER       0.67      0.89      0.76      3149
           O       0.99      0.99      0.99     42975

    accuracy                           0.93     51578
   macro avg       0.62      0.58      0.59     51578
weighted avg       0.93      0.93      0.93     51578
```

## LSTM NN classification report

### CoNLL 2003 - testa dataset

```
              precision    recall  f1-score   support

      B-MISC       0.00      0.00      0.00         4
       I-LOC       1.00      0.99      0.99    297127
      I-MISC       0.03      0.02      0.03      1264
       I-ORG       0.04      0.10      0.06      2092
       I-PER       0.09      0.12      0.10      3146
           O       0.84      0.78      0.81     42967

    accuracy                           0.95    346600
   macro avg       0.33      0.34      0.33    346600
weighted avg       0.96      0.95      0.95    346600
```

