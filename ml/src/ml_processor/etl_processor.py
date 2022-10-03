

import pandas as pd 
# import snowflake.connector

from snowflake.sqlalchemy import URL
from sqlalchemy import create_engine

from ml_processor.configuration import config
import time



class snowflake_processor:

    """

    Performing ETL tasks such as connecting to snowflake and retrieving data from snowflake.
    
    Parameters
    ----------

    username : string (default=None) 
        Username for connecting to snowflake.
        
    password : string  (default=None) 
        Password for connecting to snowflake.

    account : string (default=None)  
        Snowflake account.

    warehouse : string (default=None)  
        Warehouse name.
        
    database : string (default=None) 
        Database name.
        
    """

    def __init__(self, username=None, password=None, account=None, warehouse=None, database=None):
        
        self.credentials = config.get_credentials()
        
        self.logger = config.get_logger()
        
        if not username:
            self.username = self.credentials.get("snowflake_user")
        else:
            self.username = username
            
        if not password:
            self.password = self.credentials.get("snowflake_password")
        else:
            self.password = password
            
        if not account:
            self.account = self.credentials.get("account")
        else:
            self.account = account
            
        
        if not warehouse:
            self.warehouse = self.credentials.get("warehouse")
        else:
            self.warehouse
            
        if not database:
            self.database = self.credentials.get("database")
        else:
            self.database = database

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
        
        try:
            engine = create_engine(
                URL(
                    account = self.account,
                    user = self.username,
                    password = self.password,
                    database = self.database,
                    warehouse = self.warehouse,
                )
            )
            self.logger.info(f'Connection to {self.account} successful')
        except:
            self.logger.error('Exception occured', exc_info=True)
        else:
            return engine
        
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
            
        start = time.process_time()
            
        if not chunksize:
            
            df = pd.read_sql_query(sql, conn)
        else:
            
            df = pd.DataFrame()
            
            rows = 0
            
            for chunk in pd.read_sql_query(sql, conn, chunksize=chunksize):
                
                df = pd.concat([df, chunk])
                
                rows += chunk.shape[0]
                
        end = time.process_time()
        
        self.logger.info(f'Number of rows extracted: {df.shape[0]}')
        
        self.logger.info(f'Runtime for data extraction : {int(end-start)} seconds')
        
        return pd.DataFrame(df)