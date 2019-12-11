# Results

## CRF classification report

### CoNLL 2003

```
              precision    recall  f1-score   support

      B-MISC       0.33      0.14      0.20         7
       B-ORG       0.00      0.00      0.00         1
       I-LOC       0.88      0.89      0.89      1876
      I-MISC       0.85      0.81      0.83      1012
       I-ORG       0.83      0.81      0.82      2208
       I-PER       0.92      0.94      0.93      2349
           O       0.99      0.99      0.99     37848

    accuracy                           0.97     45301
   macro avg       0.69      0.65      0.66     45301
weighted avg       0.97      0.97      0.97     45301



performance per token:
        precision:       0.873
        recall:          0.958
        f1:              0.914


performance per named entity:
        precision:       0.867
        recall:          0.947
        f1:              0.905
```

## BI-LSTM NN classification report

### CoNLL 2003

```
              precision    recall  f1-score   support

       -PAD-       1.00      1.00      1.00    160636
      B-MISC       0.00      0.00      0.00         7
       B-ORG       0.00      0.00      0.00         1
       I-LOC       0.94      0.89      0.92      1874
      I-MISC       0.92      0.83      0.88      1012
       I-ORG       0.90      0.84      0.87      2208
       I-PER       0.97      0.87      0.91      2337
           O       0.98      1.00      0.99     37827

    accuracy                           0.99    205902
   macro avg       0.71      0.68      0.70    205902
weighted avg       0.99      0.99      0.99    205902


performance per token:
        precision:       0.934
        recall:          0.893
        f1:              0.913


performance per named entity:
        precision:       0.851
        recall:          0.959
        f1:              0.902
```

## BI-LSTM-CRF NN classification report

### CoNLL 2003

```
              precision    recall  f1-score   support

       -PAD-       1.00      1.00      1.00    160636
      B-MISC       0.50      0.14      0.22         7
       B-ORG       0.00      0.00      0.00         1
       I-LOC       0.94      0.91      0.92      1874
      I-MISC       0.74      0.87      0.80      1012
       I-ORG       0.76      0.92      0.83      2208
       I-PER       0.96      0.91      0.94      2337
           O       0.99      0.98      0.99     37827

    accuracy                           0.99    205902
   macro avg       0.74      0.72      0.71    205902
weighted avg       0.99      0.99      0.99    205902



performance per token:
        precision:       0.853
        recall:          0.963
        f1:              0.904


performance per named entity:
        precision:       0.906
        recall:          0.864
        f1:              0.884
```