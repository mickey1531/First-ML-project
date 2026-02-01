from source.MLProject.logger import logging
from source.MLProject.exception import CustomException
import sys



if __name__ == "__main__":
    logging.info("The exeuction has started")


    try:
        a = 1/0
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e, sys)
