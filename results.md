# Results

## CRF classification report

### CoNLL 2003
```
              precision    recall  f1-score   support

       B-LOC       1.00      1.00      1.00         1
      B-MISC       0.00      0.00      0.00        11
       I-LOC       0.61      0.58      0.59      1913
      I-MISC       0.72      0.63      0.67       966
       I-ORG       0.59      0.63      0.61      2133
       I-PER       0.77      0.79      0.78      2540
           O       0.99      0.99      0.99     37602

    accuracy                           0.93     45166
   macro avg       0.67      0.66      0.66     45166
weighted avg       0.93      0.93      0.93     45166
```

## LSTM NN classification report

### CoNLL 2003

```
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

       -PAD-       1.00      1.00      1.00    286799
      B-MISC       0.00      0.00      0.00         7
       B-ORG       0.00      0.00      0.00         1
       I-LOC       0.93      0.90      0.91      1876
      I-MISC       0.91      0.82      0.86      1012
       I-ORG       0.93      0.82      0.87      2208
       I-PER       0.96      0.86      0.91      2349
           O       0.98      0.99      0.99     37848

    accuracy                           1.00    332100
   macro avg       0.71      0.68      0.69    332100
weighted avg       1.00      1.00      1.00    332100
```

