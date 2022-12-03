import logging
import os
import datetime

class logger():
    _instance = None
    def __new__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance


    def __init__(self, log_dir):
        today = datetime.date.today()
        strtoday = today.strftime("%y%m%d")
        time = datetime.datetime.now()
        strhours  = time.strftime("%X").replace(":", "-")

        #cofnig logging
        log_dir = log_dir + "/" + strtoday + "/"

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_file = log_dir + strhours + ".log"
        print(log_file)
        logging.basicConfig(filename=log_file, level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logging.info("logger init")

    
    @staticmethod
    def info(message_info):
        logging.info(message_info)

    @staticmethod
    def warning(message_warning):
        logging.warning(message_warning)

    @staticmethod
    def debug(message_debug):
        logging.debug(message_debug)

    @staticmethod
    def error(message_error):
        logging.error(message_error)