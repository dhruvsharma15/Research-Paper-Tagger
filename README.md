# CS 5525: Research-Paper-Tagger
In this project, the aim is to determine the research fields of a
paper in the domain of Computer Science, from a given set of
classes. ARXIV dataset is being used for this purpose, and the multi label tags
associated with each paper is used as the classes. The novelty of
the task is that a research paper may belong to more than 1 class, so
the output will simply not be a one-hot encoded vector as generated
by any softmax classifier.
## Representaions built during the course of the project:
fastText Word Representation, Classification Model, Test file: https://drive.google.com/open?id=1l-v1rMSgC67uJH4BGMbRJwRS2BIuiUlj

Glove Representation: https://drive.google.com/open?id=1Lgym0TM-PrFBdrEz6IKDA03xjyTsxUzD

Word2Vec Representation: https://drive.google.com/open?id=1oJjjIQ0HRoUvyZC3JRhOjSrV_U6ER1cD 


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
