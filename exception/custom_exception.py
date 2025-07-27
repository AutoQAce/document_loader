import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import traceback
from logger.custom_logger import CustomLogger

logger = CustomLogger().get_logger(__file__)

class DocumentPortalException(Exception):
    """Custom exception for Document Portal"""

    def __init__(self, error_message, error_details:sys):
        _,_,exc_tb = error_details.exc_info()
        self.file_name = exc_tb.tb_frame.f_code.co_filename 
        self.line_no = exc_tb.tb_lineno
        self.error_message = error_message
        self.traceback_str = ''.join(traceback.format_exception(*error_details.exc_info()))


    def __str__(self) -> str:
        return f"""
        Error in [{self.file_name}] at line [{self.line_no}]
        Message : {self.error_message}
        Traceback:
        {self.traceback_str}
        """


if __name__ == "__main__":
    try :
        a =1/0
        print(a)
    except Exception as e:
        app_exec = DocumentPortalException(e,sys)
        logger.error(app_exec)
        raise app_exec
