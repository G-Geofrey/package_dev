

import pandas as pd 
# import snowflake.connector

from snowflake.sqlalchemy import URL
from sqlalchemy import create_engine
from cryptography.fernet import Fernet


from ml_processor.configuration import config
import time
import os
import json



class snowflake_processor:

    """

    Performing ETL tasks such as connecting to snowflake and retrieving data from snowflake.
    
    Parameters
    ----------

    credentials : dick (default=None) 
        Dictionary with connection credentials e.g
        >>> {'username':'myUserName', 'password':'******', 'account':'snowfalkeAccount', 'warehouse':'warehouseName', 'database':'dataName'}
        
    credential_path : string  (default=None) 
        Path to credentials stored in an enrypted file. This should only be provided if credentials have not been provided. 
        The contents of the file should be a dictionary encrypted using Fernet.

    key_path : string (default=None)  
        Path to Fernet key for decrypting the file provided using the credential_path..
        
    """

    def __init__(self, 
        credentials = None, 
        credential_path = None,
        key_path = None,
        ):
        
        if credentials:
            self.credentials = credentials
        else:
            if not credential_path:
                credential_path = os.path.join(os.environ["HOME"], 'Desktop/package_dev/credentials/encrypted_credentials.csv')

            if not key_path:
                key_path = os.path.join(os.environ["HOME"], 'Desktop/package_dev/credentials/sf_key.key')

            with open(key_path, 'rb') as keyfile:
                key = keyfile.read()
            
            with open(credential_path, 'rb') as credfile:
                creds = credfile.read()

            fernet = Fernet(key)

            creds = fernet.decrypt(creds).decode()

            self.credentials = json.loads(creds)

        self.logger = config.get_logger()

    def connect(self):
        
        """
        
        Create connection to snowflake.

        Parameters
        ----------

        None
        
        Returns
        -------
            
        object
            connection to snowflake.

        """
##### connection using sqlalchemy
        try:
            engine = create_engine(
                URL(
                    account = self.credentials.get('account'),
                    user = self.credentials.get('username'),
                    password = self.credentials.get('password'),
                    database = self.credentials.get('database'),
                    warehouse = self.credentials.get('warehouse'),
                    schema = self.credentials.get('schema'),
                )
            )
            self.logger.info(f'Connection to {self.credentials.get("account")} successful')
        except:
            self.logger.error('Exception occured', exc_info=True)
        else:
            return engine

##### connection using snowflake.connector
        # try:
        #     engine = snowflake.connector.connect(
        #         user = self.credentials.get('username'),
        #         password = self.credentials.get('password'),
        #         account = self.credentials.get('account'),
        #         database = self.credentials.get('database'),
        #         warehouse = self.credentials.get('warehouse'),
        #         )

        #     self.logger.info(f'Connection to {self.credentials.get("account")} successful')
        # except:
        #     self.logger.error('Exception occured', exc_info=True)
        # else:
        #     return engine

        
    def pandas_from_sql(self, sql, conn=None, chunksize=None):
        
        """

        Extracting data from snowflake into pandas dataframe.
        
        Parameters
        ----------

        sql : string 
            Sql statememt for extracting data
            
        conn : object (default=None) 
            Connection engine to snowflake.
            
        chunksize : int (default=None)
            Number of rows to extract from snowflake per iteration if extracting in chunks.
        
        
        Returns
        -------

        Pandas.DataFrame
            Data extracted fro snowflake.

        """
        
        if not conn:
            conn = self.connect()
            
        start = int(time.time())
            
        if not chunksize:
            
            df = pd.read_sql_query(sql, conn)
        else:
            
            df = pd.DataFrame()
            
            rows = 0
            
            for chunk in pd.read_sql_query(sql, conn, chunksize=chunksize):
                
                df = pd.concat([df, chunk])
                
                rows += chunk.shape[0]
                
        end = int(time.time())
        
        self.logger.info(f'Number of rows extracted: {df.shape[0]}')
        
        self.logger.info(f'Runtime for data extraction : {int(end-start)} seconds')
        
        return pd.DataFrame(df)