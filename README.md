# N2O_Emissions_Prediction

Project repository for Data Science Lab Fall'25, Project ID 1: Predicting N2O emissions from agriculture

# Repository structure

Below we provide a breakdown of the most important folders in this repository, and what they each contain:

- ```data_exploration```: our data exploration notebooks, where we do some initial data cleaning and preprocessing as well as exploratory analysis

- ```data_processing```: our notebooks for processing the data to make it ready to use in our experiments

- ```baseline_experiments```: our notebooks for the experiments with our baseline models, namely Elasticnet, Random Forest and XGBoost

- ```further_experiments```: our notebooks for our experiments beyond the baselines, where we tried different more complex models based on what we noticed worked well in our baselines

# Setup instructions

1) Firstly, you will need to ensure you have Python installed on your device. For advice on how to do that, see https://www.python.org (please ensure the Python version used is <= 3.13)

2) After that, you will need to ensure that you have Git configured on your device (see https://github.com/git-guides/install-git for details on how to install Git)

3) Once Python and git are set up, clone this repo using the following command:

```sh
git clone https://github.com/AndreasHiropedi/N2O_Emissions_Prediction.git
```

4) After cloning the repository, change directory so that you are in the repository directory using the following command:

```sh
cd N2O_Emissions_Predictions
```

5) Next, install all the necessary dependencies, namely jupyter, using the following command:

```sh
pip install jupyter
```

6) Once that is done, you will need to create a new folder called ```datasets``` inside this project directory, and you will need to place the following files in there:

- https://doi.org/10.3929/ethz-c-000782868 (rename the file to Oensingen_2021-23.csv)
- https://doi.org/10.3929/ethz-b-000747025 (rename the file to Chamau_2001-2024.csv)
- https://doi.org/10.3929/ethz-b-000584890 (rename the files to Oensingen_2018-19.csv and Aeschi_2018-19.csv)

7) Once this all is done, all the notebooks can be run as they are.
