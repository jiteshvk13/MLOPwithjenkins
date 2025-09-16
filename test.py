import sys
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

def div(x, y):
    try:
        result = x / y
        logger.info("Dividing the Numbers")
        return result
    except Exception as e:
        logger.error("Error Occurred")
        # Pass sys, not e
        raise CustomException(str(e), sys)

if __name__ == "__main__":
    try: 
        logger.info("Starting Main Program")
        div(10, 0)

    except CustomException as ce:
        logger.error(str(ce))
