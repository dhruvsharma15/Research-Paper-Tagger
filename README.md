# Research-Paper-Tagger

fastText Pre-Trained Word Vectors: https://fasttext.cc/docs/en/english-vectors.html

fastText Word Representation, Classification Model, Test file: https://drive.google.com/drive/folders/1Cp14Syec01bNa8gfgU_aNPI0-HMe1pm2?usp=sharing

## fastText Training Configuration
| Epochs        | Learning Rate | wordNgrams  |
| ------------- |:-------------:| -----------:|
| 100           | 1.0           | 2           |

Tested on 1634 examples:


| P@1        | R@1 |
| ---------- |:---:|
| 0.804      | 0.308|

To Reproduce the results, setup [fastText](https://fasttext.cc/docs/en/support.html) and run:

`./fasttext test model_da5.bin ../arxiv.pre.valid`
