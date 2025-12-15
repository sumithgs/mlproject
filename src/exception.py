import sys # used to handle exceptions
import logging

def error_message_detail(error,error_detail:sys):
    _,_,exec_tb = error_detail.exc_info() #exec_tb will contain the place file and line number where error occured.
    file_name = exec_tb.tb_frame.f_code.co_filename
    line_no = exec_tb.tb_lineno
    error_message = "Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
        file_name,line_no,str(error)
    )
    return error_message

class CustomException(Exception):
    def __init__(self, error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_detail=error_detail)
    
    def __str__(self):
        return self.error_message
    
# if __name__=="__main__":
#     try:
#         a = 1/0
#     except Exception as e:
#         logging.info("Error")
#         raise CustomException(e,sys) #- for testing