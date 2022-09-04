


from ml_processor  import config 

logger = config.get_logger()
logger.info('This works')

cred = config.get_credentials()
print(cred)