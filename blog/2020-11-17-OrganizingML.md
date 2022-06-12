---
slug: OrganizingML
title: Organizing Machine Learning Projects
tags: [Python]
authors: jack
---

**A guide to organizing your machine learning projects**

I just want to start with a brief disclaimer. This is how I personally organize my projects and it’s what works for me, that doesn't necessarily mean it will work for you. However, there is definitely something to be said about how good organization streamlines your workflow. Building my ML framework the way I do allows me to work in a very plug n’ play way: I can train, change, adjust my models without making too many changes to my code.

## Why is it important to organize a project?

1. Productivity. If you have a well-organized project, with everything in the same directory, you don't waste time searching for files, datasets, codes, models, etc.

2. Reproducibility. You’ll find that a lot of your data science projects have at least some repetitively to them. For example, with the proper organization, you could easily go back and find/use the same script to split your data into folds.

3. Understandability. A well-organized project can easily be understood by other data scientists when shared on Github.
<!--truncate-->

## File structure

The first thing I do whenever I start a new project is to make a folder and name it something appropriate, for example, “MNIST” or “digit_recognition”. Inside the main project folder, I always create the same subfolders: **notes**, **input**, **src**, **models**, **notebooks**. Don’t forget to add a `README.md` file as well! As intuitive as this breakdown seems, it makes one hell of a difference in productivity.

What do I put in each folder?

- `notes`: I add any notes to this folder, this can be anything! URLs to useful webpages, chapters/pages of books, descriptions of variables, a rough outline of the task at hand, etc.

- `input`: This folder contains all of the input files and data for the machine learning project. For example, it may contain CSV files, text embedding, or images (in another subfolder though), to list a few things.

- `src`: Any `.py` file is kept in this folder. Simple.

- `models`: We keep all of our trained models in here (the useful ones…).

- `notebooks`: We can store our Jupyter notebooks here (any `.ipynb` file). Note that I tend to only use notebooks for data exploration and plotting.

## A (basic) project example

When I build my projects I like to automate as much as possible. That is, I try to repeat myself as little as possible and I like to change things like models and hyperparameters with as little code as I can. Let's look to build a very simple model to classify the MNIST dataset. For those of you that have been living under a rock, this dataset is the de facto “hello world” of computer vision. The data files `train.csv` and `test.csv` contain gray-scale images of hand-drawn digits, from zero through nine. Given the pixel intensity values (each column represents a pixel) of an image, we aim to identify which digit the image is. This is a supervised problem. Please note that the models I am creating are by no means the best for classifying this dataset, that isn't the point of this blog post.

So, how do we start? Well, as with most things data science, we first need to decide on a metric. By looking at a count plot (`sns.countplot()` (this was done in a Jupyter notebook in the notebooks folder!)) we can see that the distribution of labels is fairly uniform, so plain and simple accuracy should do the trick!

The next thing we need to do is create some cross-validation folds. The first script I added to my src folder was to do exactly this. `create_folds.py` reads `mnist_train.csv` from the input folder and creates a new file `mnist_train_folds.csv` (saved to the input folder) which has a new column, kfolds, containing fold numbers. As with most classification problems, I used stratified k-folds. To see the script, please check out my [Github page](https://github.com/jackmleitch/Almost-any-ML-problem/tree/master/MNIST).

Now that we have decided on a metric and created folds, we can start making some basic models. With the aim of this post in mind, I decided to completely skip any kind of preprocessing (sorry!). What we are going to do is create a general python script to train our model(s), `train.py`, and then we will make some changes so that we hardcode as little as possible. The aim here is to be able to change a lot of things without changing much (if any) code. So, let's get cracking!

```py title="/src/train.py"
import joblib
import pandas as pd
from sklearn import metrics, tree
def run(fold):
    # we first read in data with our folds
    df = pd.read_csv("../input/mnist_train_folds.csv")

    # we then split it into our training and validation data
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    # drop target column (label) and turn to np array
    x_train = df_train.drop('label', axis=1).values
    y_train = df_train.label.values
    # same for validation
    x_valid = df_valid.drop('label', axis=1).values
    y_valid = df_valid.label.values
    # initialize simple decision tree classifier and fit data
    clf = tree.DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    # create predictions for validation data
    preds = clf.predict(x_valid)
    # calculate and print accuracy
    score = metrics.accuracy_score(y_valid, preds)
    print(f"Fold={fold}, Accuracy={score}")
    # save the model (not very necessary for a smaller model though)
    joblib.dump(clf, f"../models/dt_{fold}.bin")
if __name__ == "__main__":
    for i in range(3):
        run(fold=i)
```

We can then run this script by calling `python src/train.py` in the terminal. This script will read our data, train a decision tree classifier, score our predictions, and save the model for each fold. Note that you will need to set the working directory. For example:

```sh
> cd "/Users/Jack/Documents/MNIST/src"
> python train.py
Fold=0, Accuracy=0.858
Fold=1, Accuracy=0.860
Fold=2, Accuracy=0.859
```

We can do a lot better than this... We have hardcoded the fold numbers, the training file, the output folder, the model, and the hyperparameters. We can change all of this.

The first easy thing we can do is create a `config.py` file with all of the training files and output folder.

```py title="/src/config.py"
TRAINING_FILE = "../input/mnist_train_folds.csv"
OUTPUT_PATH = "../models/"
```

Changing the script to incorporate this is easy! The changes are in bold.

```py title="/src/train.py"
import os
import config
import joblib
import pandas as pd
from sklearn import metrics, tree
def run(fold):
    # we first read in data with our folds
    df = pd.read_csv(config.TRAINING_FILE)
    .
    .
    .
    # save the model
    joblib.dump(
        clf,
        os.path.join(config.OUTPUT_PATH, f"../models/dt_{fold}.bin")
    )
if __name__ == "__main__":
    for i in range(3):
        run(fold=i)
```

In our script, we call the **run** function for each fold. When training bigger models this can be an issue as running multiple folds in the same script will keep increasing the memory consumption. This could potentially lead to the program crashing. We can get around this by using **argparse**. Argparse allows us to specify arguments in the command line which get parsed through to the script. Let's see how to implement this.

```py title="/src/train.py"
import argparse
.
.
.
if __name__ == "__main__":
    # initialize Argument Parser
    parser = argparse.ArgumentParser()

    # we can add any different arguments we want to parse
    parser.add_argument("--fold", type=int)
    # we then read the arguments from the comand line
    args = parser.parse_args()
    # run the fold that we specified in the command line
    run(fold=args.fold)
```

So what have we done here? We have allowed us to specify the fold in the terminal.

```sh
> python train.py --fold 3
Fold=3, Accuracy=0.861
```

We can use argparse in an even more useful way though, we can change the model with this! We can create a new python script, `model_dispatcher.py`, that has a dictionary containing different models. In this dictionary, the keys are the names of the models and the values are the models themselves.

```py title="/src/model_dispatcher.py"
from sklearn import tree, ensemble, linear_model, svm
models = {
     "decision_tree_gini": tree.DecisionTreeClassifier(
         criterion="gini"
     ),
     "decision_tree_entropy": tree.DecisionTreeClassifier(
         criterion='entropy'
     ),
     "rf": ensemble.RandomForestClassifier(),
     "log_reg": linear_model.LogisticRegression(),
     "svc": svm.SVC(C=10, gamma=0.001, kernel="rbf")
}
```

We can then add the following things to our code.

```py title="/src/train.py"
import os, argparse, joblib
import pandas as pd
from sklearn import metrics
import config
import model_dispatcher
def run(fold, model):
    # we first read in data with our folds
    df = pd.read_csv(config.TRAINING_FILE)
    # we then split it into our training and validation data
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    # drop target column (label) and turn to np array
    x_train = df_train.drop('label', axis=1).values
    y_train = df_train.label.values
    x_valid = df_valid.drop('label', axis=1).values
    y_valid = df_valid.label.values
    # fetch model from model dispatcher
    clf = model_dispatcher.models[model]
    # fit model on training data
    clf.fit(x_train, y_train)
    # create predictions for validation data
    preds = clf.predict(x_valid)
    # calculate and print accuracy
    score = metrics.accuracy_score(y_valid, preds)
    print(f"Model={model}, Fold={fold}, Accuracy={score}")
    # save the model
    joblib.dump(
        clf,
        os.path.join(config.OUTPUT_PATH, f"../models/dt_{fold}.bin")
    )
if __name__ == "__main__":
  # initialize ArgumentParser class of argparse
  parser = argparse.ArgumentParser()

  # add different arguments and their types
  parser.add_argument('--fold', type=int)
  parser.add_argument('--model', type=str)

  # read arguments from command line
  args = parser.parse_args()
  # run with the fold and model specified by command line arguments
  run(
      fold=args.fold,
      model=args.model
  )
```

And that's it, now we can choose the fold and model in the terminal! What's great about this is that to try a new model/tweak hyperparameters, all we need to do is change our model dispatcher.

```sh
> python train.py --fold 2 --model rf
  Model=rf, Fold=2, Accuracy=0.9658333333333333
```

We also aren’t limited to just doing this. We can make dispatchers for loads of other things too: categorical encoders, feature selection, hyperparameter optimization, the list goes on!

To go even further, we can create a shell script to try a few different models from our dispatcher at once.

```sh title="/src/run.sh"
#!/bin/sh
python train.py --fold 0 --model rf
python train.py --fold 0 --model decision_tree_gini
python train.py --fold 0 --model log_reg
python train.py --fold 0 --model svc
```

Which, when run, gives:

```sh
> sh run.sh
Model=rf, Fold=0, Accuracy=0.9661666666666666
Model=decision_tree_gini, Fold=0, Accuracy=0.8658333333333333
Model=log_reg, Fold=0, Accuracy=0.917
Model=svc, Fold=0, Accuracy=0.9671812654676666
```

I first learned how to do all of this in Abhishek Thakur’s (quadruple Kaggle Grand Master) book: Approaching (Almost) Any Machine Learning Problem. It’s a fantastic and pragmatic exploration of data science problems. I would highly recommend it.

## A word on Github

Once you’ve finished up (or during!) a problem that you find interesting make sure to create a Github repository and upload your datasets, python scripts, models, Jupyter notebooks, R scripts, etc. Creating Github repositories to showcase your work is extremely important! It also means you have some version control and you can access your code at all times.
