# Disaster Response Messages Classification

In this project, we create a web application that classify disaster messages.

For more detail about the dataset used in the project, go to section [Dataset](#Dataset).

> For the newer version of dataset see [here](https://appen.com/datasets/combined-disaster-response-data/).

## Getting Started

### Requirements

This repository uses Python 3.8.

* We use [pyenv](https://github.com/pyenv/pyenv) and [pipenv](https://github.com/pypa/pipenv) to manage the python version and environment.
* Please see `Pipfile` for package and its versions used in this project.


### Preparation

* Clone this repository
```bash
git clone https://github.com/syahrulhamdani/disaster-response-pipeline
```
* We use pipenv to manage the environment. To install all libraries in Pipfile,
run below

```bash
PIPENV_VENV_IN_PROJECT=1 pipenv install --dev
```

* **IMPORTANT**. Make sure to create your `.env` for necessary environment variables.
`.env.example` could be your template, just copy paste it.


## Usage

1. Make sure you have the dataset, `disaster_messages.csv` and `disaster_categories.csv`, inside `data`
directory.

2. Perform ETL steps to process both data by running

```bash
pipenv run python disaster_messages.csv disaster_categories.csv disaster.db
```

This will process and transform both dataset and load it in a SQL database `disaster.db`.

3. Now that the dataset is ready, build and train the model by running

```bash
pipenv run python train.py disaster.db model.joblib
```

This will save the model in `models/model.joblib`.

4. As all is ready, we can run the web app in our local environment using below command

```bash
pipenv run python server.py
```

## Dataset

* There are about categories with each message can belong to more than 1 category.
Hence, it's multilabel data.
* The dataset is imbalanced which each category has a contrast number of `True` and `False` value.

![](viz/message_categories.png)
