# Results

## CRF classification report

### CoNLL 2003
```
              precision    recall  f1-score   support

      B-MISC       1.00      0.14      0.25         7
       B-ORG       0.00      0.00      0.00         1
       I-LOC       0.86      0.85      0.86      1876
      I-MISC       0.84      0.82      0.83      1012
       I-ORG       0.78      0.77      0.77      2208
       I-PER       0.89      0.92      0.91      2349
           O       0.99      0.99      0.99     37848

    accuracy                           0.97     45301
   macro avg       0.77      0.64      0.66     45301
weighted avg       0.97      0.97      0.97     45301
```

## BI-LSTM NN classification report

### CoNLL 2003

```
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

## BI-LSTM-CRF NN classification report

```
              precision    recall  f1-score   support

       -PAD-       1.00      1.00      1.00    160636
      B-MISC       0.00      0.00      0.00         7
       B-ORG       0.00      0.00      0.00         1
       I-LOC       0.88      0.83      0.86      1874
      I-MISC       0.91      0.61      0.73      1012
       I-ORG       0.83      0.68      0.75      2208
       I-PER       0.90      0.84      0.87      2337
           O       0.97      1.00      0.98     37827

    accuracy                           0.99    205902
   macro avg       0.69      0.62      0.65    205902
weighted avg       0.99      0.99      0.99    205902
```
