### OSX/Linux Virtual Environment Setup (Python)

#### Setting up a virtual environment & installing requirements.txt packages

We highly recommend setting up a virtual environment before running the `run_tester_python.py` script.

The main purpose of a Python virtual environment is to create an isolated environment for Python projects. You can find more infomation [here](https://docs.python-guide.org/dev/virtualenvs/).

Run the following commands from within the `python` directory to setup a Python virtual environment:

#### install virtualenv
```console
$ python3 -m pip install --user virtualenv
```

#### creating a virtual environment
```console
$ python3 -m venv env
```

#### activating a virtual environment
```console
$ source env/bin/activate
```

#### install packages from requirements.txt
```console
$ pip3 install -r requirements.txt
```
