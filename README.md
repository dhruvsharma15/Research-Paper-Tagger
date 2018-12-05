# CS 5525: Research-Paper-Tagger
With the exponential growth of data, conceptual organization of document collection via text classification has become more relevant. One of the interesting applications of text classification is to categorize research papers by the relevant academic areas it belongs to. Finding the relevant tags corresponding to a scientific paper and a suitable academic conference which is aligned with the researcher's work poses a challenge for many students and young researchers. In many existing automatic methods to solve this problem, a naive strategy is to do a keyword-based search using the paper's title. However, this approach fails due to the presence of semantically equivalent terms but not the same subject words. In this project, the aim is to perform a multi-label text classification to determine the research fields of a paper in the domain of Computer Science. ARXIV dataset is being used for this purpose, and the areas associated with each paper is used as the classes. The novelty of the task is that a research paper may belong to more than one class, so the output will simply not be a one-hot encoded vector as generated by any softmax classifier. Lastly, we extend our project propsal by building a recommender system which outputs similar reseach papers, provided a scientific text.
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
