import pandas as pd 
import snowflake.connector

from  ml_processor.configuration import config



class snowflake_processor:
    """
    Class for performing ETL tasks such as connecting to snowflake and retrieving data from snowflake
    
    Attributes:
        username (string) username for connecting to snowflake
        password (string) password for connecting to snowflake
        account (string) name of snowflake account
        warehouse (string) warehouse name
        database (string) database name
        
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
        Function for creating connection to snowflake
        
        Returns:
            object: connection to snowflake
        """
        
        try:
            conn = snowflake.connector.connect(user=self.username,
                                       password=self.password,
                                       account=self.account,
                                       warehouse=self.warehouse,
                                       database=self.database      
                                       )
            self.logger.info(f'Connection to {self.account} successful')
        except:
            self.logger.error('Exception occured', exc_info=True)
        else:
            return conn
        
    def pandas_from_sql(self, sql, conn=None, chunksize=None):
        
        """
        Function for extracting data from snowflake into pandas dataframe
        
        Args:
            
            sql (string) sql statememt for extracting data
            
            conn (object) connection to snowflake
            
            chunksize (int) number of rows to extract from snowflake per iteration if extracting in chunks
        
        
        Returns:
            Dataframe: Data extracted fro snowfale
        """
        
        if not conn:
            conn = self.connect()
        
        if not chunksize:
            df = pd.read_sql_query(sql, conn)
        else:
            df = pd.DataFrame()
            rows = 0
            for chunk in pd.read_sql_query(sql, conn, chunksize=chunksize):
                df = pd.concat([df, chunk])
                rows += chunk.shape[0]
        self.logger.info(f'Number of rows extracted: {df.shape[0]}')
        
        return df