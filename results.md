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

