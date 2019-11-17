### Windows Virtual Environment Setup (Python)

#### Setting up a virtual environment & installing requirements.txt packages

We highly recommend setting up a virtual environment before running the `run_tester_python.py` script.

The main purpose of a Python virtual environment is to create an isolated environment for Python projects. You can find more infomation [here](https://docs.python-guide.org/dev/virtualenvs/).

Run the following commands from within the `python` directory to setup a Python virtual environment:

#### Installing `virtualenv`
```console
$ py -m pip install --user virtualenv
```

#### Creating a virtual environment
```console
$ py -m venv env
```

#### Activating a virtual environment
```console
$ .\env\Scripts\activate
```

#### Installing packages from `requirements.txt`
```console
$ py -m pip install -r requirements.txt
```
