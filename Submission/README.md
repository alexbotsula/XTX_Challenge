# XTX Markets Forecasting Challenge StarterKit

## Instructions

The goal of this challenge is to create a program to predict stock movement based on real-market orderbook data. A dataset and some starter code has been provided to help build submissions.

## data.csv
* A small sample of the dataset is included in this starter kit. The full dataset `data-training.csv` can be downloaded from https://challenge.xtxmarkets.com/ under the `Data Download` section.
* After downloading the dataset, it can be included in the root of the StarterKit directory, but it **should not** be included in the final submission.
* This file contains some data to be analyzed and trained with the submitted model. 
* The goal of the challenge is to predict the `y` values (the column on the far right), given the current and all past rows. 
* The submitted model will be tested on data which is similar to `data.csv`; however it will not contain the `y` column.

### StarterKit Folder Structure

There are two programming languages that are currently supported for this challenge, Python and R.

In this challenge, There are three relevant folders - one named `src` containing all the source code needed to process data 
files and getting your result, another two named either `python` or `r` depending on preference of language to be used in the 
submission. The first step should be to choose a programming language by removing the folder that is unnecessary for the 
submission. For example, if the language used in a submission is Python, the `r` folder should be removed.

### Quickstart

* Submissions will not be able to access the Internet on our submission environment. 

## OSX and Linux

* For **Python** submissions on `OSX` or `Linux` see the docs under [docs/quickstart_osx_linux/python.md](docs/quickstart_osx_linux/python.md).
* For **R** submissions on `OSX` or `Linux` see the docs under [docs/quickstart_osx_linux/r.md](docs/quickstart_osx_linux/r.md).
* We **highly** recommend using a virtual environment for Python development: [docs/quickstart_osx_linux/virtual_env.md](docs/quickstart_osx_linux/virtual_env.md).

## Windows

* For **Python** submissions on `Windows` see the docs under [docs/quickstart_windows/python.md](docs/quickstart_windows/python.md).
* For **R** submissions on `Windows` see the docs under [docs/quickstart_windows/r.md](docs/quickstart_windows/r.md).
* We **highly** recommend using a virtual environment for Python development: [docs/quickstart_windows/virtual_env.md](docs/quickstart_windows/virtual_env.md)

### The `src` folder

**None of these files require changing**; however, there are a couple of files in this folder that are quite important to know.
**Please refrain from trying to run these files manually**; there is a script in the `python` and R folders that runs the model_tester and scorer.

## `src/model_tester.py`
* A prediction vector will be outputted at the value of `RESULT_LOCATION`.
* Any changes made to this file will not persist on the testing servers.

## `src/scorer.py`
* This program is used to score a prediction vector.
* This program will print the final `r2` value to stdout after running.

### The Submission Folders

#### `python`

This folder should be removed if the solution is written in R.

##### `python/core.py`
* This file does not need to be modified
* This file contains the `Submission` class, all code to interact with `src/model_tester.py`

##### `python/submission.py`
* This file should be used if the solution is written in Python.
* This file contains the starter code for the submission program.
* This extends the `Submission` class found in `core.py`.
* The function `self.get_next_data()` **must** be used to read a line of data.
* The function `self.submit_prediction(pred)` **must** be used to submit a prediction for the `y` value of the next row of data.
* **`self.get_next_data()` cannot be called two or more times in a row without calling `self.submit_prediction(pred)`**.
* Messages should not be printed to stdout because `model_tester.py` will be looking for predictions from stdout.
* To debug, messages should be printed to `stderr`.

##### `python/requirements.txt`
* **Any packages or dependencies necessary for the submission should be added here.**
* These will be installed at runtime.

#### `r`

This folder should be removed if the solution is written in Python.

##### `r.submission.r`
* This file should be used if the solution is written in R.
* This file contains the starter code for the submission program.
* The function `get_next_data()` **must** be used to read a line of data.
* The function `submit_prediction(pred)` **must** be used to submit a prediction for the `y` value of the next row of data.
* **`get_next_data()` cannot be called two or more times in a row without calling `submit_prediction(pred)`**.
* Messages should not be printed to stdout because `model_tester.py` will be looking for predictions from stdout.
* To debug, messages should be printed to stderr.

##### `r/requirements.txt`
* **Any packages or dependencies necessary for the R submission should be added here.**
* These will be installed at runtime.

### Submission Instructions

#### For Python submissions

* Follow the steps in [docs/submissions/python.md](docs/submissions/python.md) to upload a Python submission.

#### For R submissions

* Follow the steps in [docs/submissions/r.md](docs/submissions/r.md) to upload a R submission.
