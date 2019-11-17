### Python Quickstart (macOS and Linux)

Requirements: Python 3.6 or higher

Files that are required for your submission to run successfully such as: models, files that you create must be stored in `python`.

Place required Python packages in the `requirements.txt` file. These packages will be installed on the fly on our submission environment.

Our testing script `run_tester_python.py` will **NOT** install these packages; therefore, please ensure your system has the packages installed in order to test locally.

At all phases of the development process it is highly recommended to run `python3 run_tester_python.py` from within the `python` directory.

If your submission is not able to run with `python3 run_tester_python.py`, it will NOT run on our platform.

This script will ensure that the submission folder satisfies:  

|--README.md<br />
|-- python<br />
---|-- core.py<br />
---|-- requirements.txt<br />
---|-- submission.py<br />
---|-- run_tester_python.py<br />
|-- src<br />
---|-- model_tester.py<br />
---|-- scorer.py<br />
|-- data.csv<br />

This script `python3 run_tester_python.py` will also run `src/model_tester.py`, and return a score. 

The result & score can be found in the results folder that will be created upon successfully running `python3 run_tester_python.py`.
