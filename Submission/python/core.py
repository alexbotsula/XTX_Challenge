import sys, os, math

class GetNextDataCalledTwiceException(Exception):
    pass

class NotInstalledPackageException(Exception):
    pass

def enable_print():
    sys.stdout = sys.__stdout__

def disable_print():
    sys.stdout = open(os.devnull, 'w')

class Submission():
    def __init__(self):
        try:
            disable_print()
            self.DATA_ROW_IN_TRANSIT = False
            self.run_submission()
        except EOFError as e:
            pass

    def run_submission(self):
        raise NotImplementedError("Please implement run_submission in your " +
                "submission class")

    def get_next_data_as_string(self):
        """
        Reads input from standard input

        Use this to supply your model with input
        Input will not be supplied until output is 
        generated for the previous input
        """

        if self.DATA_ROW_IN_TRANSIT:
            raise GetNextDataCalledTwiceException("get_next_data_as_string() can only be called once for every prediction made.")
        
        data_row = input()
        self.DATA_ROW_IN_TRANSIT = True
        return data_row
        
    def get_next_data_as_list(self):
        """
        Reads input from standard input and stores row in a 
        list where missing values are represented as NaN

        Use this to supply your model with input
        Input will not be supplied until output is 
        generated for the previous input
        """

        if self.DATA_ROW_IN_TRANSIT:
            raise GetNextDataCalledTwiceException("get_next_data_as_list() can only be called once for every prediction made.")
        

        raw_data_list = input().split(",")
    
        # replace empty spots with NaN
        data_list = []
        for order in raw_data_list:
            if not order:
                data_list.append(math.nan)
            else:
                data_list.append(float(order))
        
        self.DATA_ROW_IN_TRANSIT = True
        return data_list
    
    def get_next_data_as_numpy_array(self):
        """
        Reads input from standard input and stores row in a
        numpy array where missing values are represented as NaN

        Use this to supply your model with input
        Input will not be supplied until output is 
        generated for the previous input
        """

        if self.DATA_ROW_IN_TRANSIT:
            raise GetNextDataCalledTwiceException("get_next_data_as_numpy_array() can only be called once for every prediction made.")

        import numpy
        return numpy.array(self.get_next_data_as_list())

    def submit_prediction(self, prediction):
        """
        Submits your prediction to standard output
        """
        enable_print()
        print(str(prediction))
        sys.stdout.flush()
        disable_print()

        self.DATA_ROW_IN_TRANSIT = False
        

    def debug_print(self, msg):
        """
        Prints to standard error

        Use this to debug / develop. 
        This output will not be scored
        """
        print(msg, file=sys.stderr)


