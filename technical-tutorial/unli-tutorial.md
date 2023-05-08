# UNLI: Regression modeling with Transformers

Natural Language Processing (NLP) is a task where the goal is to determine whether a given statement (the "hypothesis") is entailed by or contradicts another statement (the "premise"). It is usually labeled as categories, such as "entailment", "contradiction" and "neutral". Traditionally, NLI has been approached as a categorical classification problem where the labels are coarse-grained and not able to capture the uncertain and probabilistic nature of entailment inference. In other words, not all contradictions or entailments are equally strong or certain, and assigning them to coarse categories may hide this variability. For example, some entailments may be highly probable, while others may be only slightly probable.

Example:

## Premise -> Hypothesis -> NLI -> UNLI

A man in a white shirt taking a picture -> A man takes a picture -> ENT -> 100%

A boy hits a ball, with a bat -> The kid is playing in a baseball game -> ENT -> 78%

A wrestler in red cries, one in blue celebrates -> The wrestler in blue is undefeated -> CON -> 50%

Man laying on a platform outside on rocks -> Man takes a nap on his couch -> CON -> 0%

[UNLI](https://github.com/clips/wordkit) (Uncertain Natural Language Inference) is a new approach to Natural Language Inference that aims to predict the probability of a hypothesis being true, instead of simply assigning a categorical label like in NLI. 

In this tutorial, we will learn to create a regression model using transformers.


*Let's get started!*

## Installing requirements:

* transformers
* datasets
* pandas
* numpy
* scikit-learn


## Data
A dataset called u-SNLI was created by labeling a portion of the widely-used SNLI dataset using a probabilistic scale. The u-SNLI dataset is available for download [here](https://nlp.jhu.edu/unli/) and includes three files: train.csv, dev.csv, and test.csv. Each file contains columns for ID, Premise, Hypothesis, NLI label, and UNLI label.

Example annotations:

* Premise: A man is singing into a microphone.
* Hypotheses:

[0.946] A man performs a song.

[0.840] A man is performing on stage.

[0.152] A male performer is singing a special and meaningful song.

[0.144] A man performing in a bar.

[0.062] A man is singing the national anthem at a crowded stadium.

## Imports

We use the following in this tutorial:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from transformers import (
    AdamW, 
    AutoTokenizer, 
    AutoModel, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments, 
    DataCollatorWithPadding,
)

from datasets import load_dataset, DatasetDict
```

### Let's Begin!

## 1. Reading in the corpora

A zip file of the u-snli dataset can be downloaded as mentioned above and can be stored and accessed in the most suitable way. For ease, I store it in a google drive and read it as follows:
```python
from google.colab import drive

drive.mount('/content/drive', force_remount=True)
```

## 2. Setting a list of training sizes 

The list of training sizes is used to explore how the performance of the model changes as the amount of training data increases. By training the model on different amounts of data, we can observe how the model's learning curve evolves. For the purpose of the tuyorial, the training sizes are:
```python
training_size = [15, 30]
```

## 3. Loading data

I load the data as follows:

```python
train = load_dataset('csv', data_files='path_to_u-snli_train.csv')

test = load_dataset('csv', data_files='path_to_u-snli_test.csv')

validation = load_dataset('csv', data_files='path_to_u-snli_dev.csv') 
```

## 4. Definig the Model and Tokenizer

The following code block uses the Hugging Face Transformers library to load a pre-trained BERT model and its corresponding tokenizer.

The checkpoint variable is set to "bert-base-uncased", which is the name of a pre-trained BERT model provided by Hugging Face. The AutoTokenizer.from_pretrained() function is then called with the checkpoint parameter to create a tokenizer that can tokenize text input into the format expected by the BERT model.

The AutoModelForSequenceClassification.from_pretrained() function is called with checkpoint and num_labels=1 parameters to load the pre-trained BERT model and initialize it for sequence classification tasks with a single output label. This will be useful for the current task of predicting a single numerical value as the output.
```python
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=1) 
```

## 5. Tokenizing the data

The tokenization function is defined to tokenize the text data using the BERT tokenizer. The function also sets the label for each example in the data.
```python
def tokenize_function(examples):
    label = examples["unli"] 
    examples = tokenizer(examples["pre"], examples["hyp"], truncation=True, padding="max_length", max_length=256)
    
    
    examples["label"] = label
    return examples
```

## 6. Defining the training arguments

The training arguments are defined using the TrainingArguments class provided by the Transformers library. The arguments include the learning rate, batch size, number of epochs, evaluation strategy, and model saving strategy.
```python
LEARNING_RATE = 1e-5
MAX_LENGTH = 256
BATCH_SIZE = 8
EPOCHS = 2

training_args = TrainingArguments(
  output_dir="Path_to_u-snli_results",
  learning_rate=LEARNING_RATE,
  per_device_train_batch_size=BATCH_SIZE,
  per_device_eval_batch_size=BATCH_SIZE,
  num_train_epochs=EPOCHS,
  evaluation_strategy="epoch",
  save_strategy="epoch",
  save_total_limit=2,
  metric_for_best_model="mse",
  load_best_model_at_end=True,
  weight_decay=0.01,
)

```

## 7. Training the model

A loop is created to train the model on different sizes of training data (15 and 30 in this case). The training data is tokenized and passed to the RegressionTrainer class provided by the Transformers library to train the model on the data.
```python
  trainer = RegressionTrainer(
      model=model,
      args=training_args,
      train_dataset=ds[0],
      eval_dataset=ds[1],
      compute_metrics=compute_metrics_for_regression,
  )

  trainer.train()
```

## 8. Evaluating the model

After each training iteration, the model is evaluated on the validation and test datasets using the evaluate() method of the RegressionTrainer class. The evaluation metrics are calculated using the compute_metrics_for_regression() function defined as follows:
```python
def compute_metrics_for_regression(eval_pred):
  logits, labels = eval_pred
  labels = labels.reshape(-1, 1)
      
  mse = mean_squared_error(labels, logits)
  mae = mean_absolute_error(labels, logits)
  r2 = r2_score(labels, logits)
      
  return {"mse": mse, "mae": mae, "rmse": np.sqrt(mse), "r2": r2}
```

## 9. Plotting the learning curve

Finally, we can now plot a graph to see the learning curve of the model with respect to the mean squared error (MSE) metric for different training sizes. The following code shows an example for a different training size and mse values:
```python
import matplotlib.pyplot as plt
training_size = [10, 15, 20, 25, 30]
mse = [0.09804137051105499, 0.08982931077480316, 0.09684375673532486, 0.08992088586091995, 0.08983176201581955]
print("Learning Curve for MSE \n")  
plt.plot(training_size, mse)
plt.xlabel('Train/Validation')
plt.ylabel('MSE')
plt.show()
```

