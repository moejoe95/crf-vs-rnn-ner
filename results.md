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

       -PAD-       1.00      1.00      1.00    286902
       B-LOC       0.00      0.00      0.00         1
      B-MISC       0.00      0.00      0.00        14
       I-LOC       0.92      0.87      0.90      1902
      I-MISC       0.85      0.79      0.82      1080
       I-ORG       0.81      0.87      0.84      2189
       I-PER       0.92      0.93      0.92      2486
           O       0.99      0.99      0.99     37526

    accuracy                           1.00    332100
   macro avg       0.69      0.68      0.68    332100
weighted avg       1.00      1.00      1.00    332100
```

