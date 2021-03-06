# Results

## CRF classification report

Evaluation is done via the python script `conlleval.py` from https://github.com/spyysalo/conlleval.py. 
Each test was also done with pretrained embeddings from [glove](https://nlp.stanford.edu/projects/glove/) with a vector size of 100.

### CoNLL 2003

```
processed 45301 tokens with 5047 phrases; found: 5117 phrases; correct: 4311.
accuracy:  97.30%; precision:  84.25%; recall:  85.42%; FB1:  84.83
              LOC: precision:  89.40%; recall:  88.73%; FB1:  89.06  1594
             MISC: precision:  77.98%; recall:  83.29%; FB1:  80.55  754
              ORG: precision:  76.25%; recall:  79.49%; FB1:  77.84  1377
              PER: precision:  89.66%; recall:  88.26%; FB1:  88.95  1392
```

### W-NUT 17

```
processed 12876 tokens with 268 phrases; found: 482 phrases; correct: 152.
accuracy:  94.92%; precision:  31.54%; recall:  56.72%; FB1:  40.53
      corporation: precision:  29.41%; recall:  52.63%; FB1:  37.74  34
    creative-work: precision:  13.04%; recall:  37.50%; FB1:  19.35  46
            group: precision:   3.75%; recall:  23.08%; FB1:   6.45  80
         location: precision:  37.76%; recall:  60.66%; FB1:  46.54  98
           person: precision:  50.00%; recall:  63.70%; FB1:  56.02  186
          product: precision:   7.89%; recall:  23.08%; FB1:  11.76  38
```

## BI-LSTM NN classification report

### CoNLL 2003

```
processed 45257 tokens with 4872 phrases; found: 5110 phrases; correct: 4243.
accuracy:  97.32%; precision:  83.03%; recall:  87.09%; FB1:  85.01
              LOC: precision:  89.14%; recall:  92.57%; FB1:  90.82  1593
             MISC: precision:  80.64%; recall:  86.73%; FB1:  83.57  754
              ORG: precision:  79.23%; recall:  80.04%; FB1:  79.64  1377
              PER: precision:  81.10%; recall:  88.23%; FB1:  84.51  1386
```

With pretrained embeddings:
```
processed 45156 tokens with 4963 phrases; found: 5093 phrases; correct: 4297.
accuracy:  97.49%; precision:  84.37%; recall:  86.58%; FB1:  85.46
              LOC: precision:  89.18%; recall:  92.80%; FB1:  90.95  1589
             MISC: precision:  81.96%; recall:  85.71%; FB1:  83.80  754
              ORG: precision:  82.02%; recall:  79.46%; FB1:  80.72  1368
              PER: precision:  82.49%; recall:  87.49%; FB1:  84.92  1382
```

### W-NUT 17

```
processed 12762 tokens with 177 phrases; found: 472 phrases; correct: 41.
accuracy:  94.37%; precision:   8.69%; recall:  23.16%; FB1:  12.63
      corporation: precision:   0.00%; recall:   0.00%; FB1:   0.00  34
    creative-work: precision:   0.00%; recall:   0.00%; FB1:   0.00  45
            group: precision:   0.00%; recall:   0.00%; FB1:   0.00  79
         location: precision:   6.32%; recall:  60.00%; FB1:  11.43  95
           person: precision:  19.34%; recall:  21.21%; FB1:  20.23  181
          product: precision:   0.00%; recall:   0.00%; FB1:   0.00  38
```

With pretrained embeddings:
```
processed 12477 tokens with 163 phrases; found: 463 phrases; correct: 28.
accuracy:  94.25%; precision:   6.05%; recall:  17.18%; FB1:   8.95
      corporation: precision:   0.00%; recall:   0.00%; FB1:   0.00  33
    creative-work: precision:   0.00%; recall:   0.00%; FB1:   0.00  46
            group: precision:   0.00%; recall:   0.00%; FB1:   0.00  78
         location: precision:   2.17%; recall:  18.18%; FB1:   3.88  92
           person: precision:  14.44%; recall:  17.22%; FB1:  15.71  180
          product: precision:   0.00%; recall:   0.00%; FB1:   0.00  34
```

## BI-LSTM-CRF NN classification report

### CoNLL 2003

```
processed 45266 tokens with 5297 phrases; found: 5110 phrases; correct: 4417.
accuracy:  96.96%; precision:  86.44%; recall:  83.39%; FB1:  84.89
              LOC: precision:  90.90%; recall:  90.11%; FB1:  90.50  1593
             MISC: precision:  83.16%; recall:  65.59%; FB1:  73.33  754
              ORG: precision:  83.15%; recall:  80.69%; FB1:  81.90  1377
              PER: precision:  86.36%; recall:  91.03%; FB1:  88.63  1386
```

With pretrained embeddings:

```
processed 45265 tokens with 5299 phrases; found: 5109 phrases; correct: 4568.
accuracy:  97.62%; precision:  89.41%; recall:  86.20%; FB1:  87.78
              LOC: precision:  94.79%; recall:  87.69%; FB1:  91.10  1593
             MISC: precision:  86.34%; recall:  76.05%; FB1:  80.87  754
              ORG: precision:  84.88%; recall:  82.14%; FB1:  83.49  1376
              PER: precision:  89.39%; recall:  95.38%; FB1:  92.29  1386
```

### W-NUT 17

```
processed 12876 tokens with 750 phrases; found: 482 phrases; correct: 135.
accuracy:  90.63%; precision:  28.01%; recall:  18.00%; FB1:  21.92
      corporation: precision:  44.12%; recall:  55.56%; FB1:  49.18  34
    creative-work: precision:  17.39%; recall:  20.00%; FB1:  18.60  46
            group: precision:  27.50%; recall:   5.84%; FB1:   9.63  80
         location: precision:  31.63%; recall:  31.63%; FB1:  31.63  98
           person: precision:  29.03%; recall:  40.30%; FB1:  33.75  186
          product: precision:  13.16%; recall:   6.76%; FB1:   8.93  38
```

With pretrained embeddings:
```
processed 12876 tokens with 522 phrases; found: 482 phrases; correct: 162.
accuracy:  93.43%; precision:  33.61%; recall:  31.03%; FB1:  32.27
      corporation: precision:  44.12%; recall:  55.56%; FB1:  49.18  34
    creative-work: precision:  26.09%; recall:  14.63%; FB1:  18.75  46
            group: precision:  32.50%; recall:  14.77%; FB1:  20.31  80
         location: precision:  39.80%; recall:  41.49%; FB1:  40.62  98
           person: precision:  34.41%; recall:  57.14%; FB1:  42.95  186
          product: precision:  15.79%; recall:  19.35%; FB1:  17.39  38
```
