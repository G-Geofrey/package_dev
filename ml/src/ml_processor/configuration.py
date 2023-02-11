
import logging
import sys
# from dotenv import dotenv_values 
import json
import numpy as np
import os



class config: 
        
    # def add_path(lib_path=None):

    #     """

    #     Add path to working path

    #     Parameters
    #     ----------

    #     lib_path : string (default=None)
    #         Path to add to working path

    #     Returns
    #     -------

    #     None

    #     """
    #     if lib_path:
    #         sys.path.insert(1, lib_path)
    #     else:
    #         lib_path = os.path.join(os.environ.get("HOME"), "Library/CloudStorage/OneDrive-MoneseLtd/GW/libraries/snowflake")
    #         sys.path.insert(1, lib_path)  
    
    # def get_credentials(env_path=None):

    #     """
        
    #     Get credentials stored in dot file

    #     Parameters
    #     ----------
        
    #     env_path : string (deault=None)
    #         Path to dot file with credentials

    #     Returns
    #     -------

    #     Dic:
    #         Stored credentials

    #     """
        
    #     if env_path:
    #         env_path = env_path
    #     else:
    #         env_path = os.path.join(os.environ.get("HOME"), "Library/CloudStorage/OneDrive-MoneseLtd/GW/libraries/environ_vars/.env")
            
    #     return dotenv_values(env_path)
    
    def get_logger(file=None, folder="model_artefacts"):

        """

        Create logger

        Parameters
        ----------

        file : string (default=None)

        Returns
        -------

        object:
            Logger

        """

        logger = logging.getLogger('logger_name')

        logger.setLevel(logging.INFO)

        logger.handlers = []

        formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")

        streamhandler = logging.StreamHandler(stream=sys.stdout)

        streamhandler.setLevel(logging.DEBUG)

        streamhandler.setFormatter(formatter)

        logger.addHandler(streamhandler)

        if file:

            if not os.path.isdir(f'{ folder}'):
                
                os.mkdir(f'{folder}')

            path = os.path.join(f'{folder}', file)

            filehandler = logging.FileHandler(path)

            filehandler.setLevel(logging.DEBUG)

            filehandler.setFormatter(formatter)

            logger.addHandler(filehandler)

        logger.propagate=False

        return logger


        