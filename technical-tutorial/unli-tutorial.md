# UNLI: Regression modeling with Transformers

[UNLI](https://github.com/clips/wordkit) (Uncertain Natural Language Inference)is a new approach to Natural Language Inference that aims to predict the probability of a hypothesis being true, instead of simply assigning a categorical label like in NLI. 


*Let's get started!*

## Installing WordKit
You can install WordKit with pip on the command line using `pip install wordkit`. WordKit has the following requirements:
* ipapy
* numpy
* pandas
* reach
* nltk
* scikit-learn


## Data
A dataset called u-SNLI was created by labeling a portion of the widely-used SNLI dataset using a probabilistic scale. The u-SNLI dataset is available for download [here](https://nlp.jhu.edu/unli/)and includes three files: train.csv, dev.csv, and test.csv. Each file contains columns for ID, Premise, Hypothesis, NLI label, and UNLI label.

Example annotations:

*Premise: A man is singing into a microphone.
*Hypotheses:
[0.946] A man performs a song.
[0.840] A man is performing on stage.
[0.152] A male performer is singing a special and meaningful song.
[0.144] A man performing in a bar.
[0.062] A man is singing the national anthem at a crowded stadium.
