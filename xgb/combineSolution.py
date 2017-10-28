import pandas as pd

PRIVATE_EXPERIMENT_NUMBER = 1
HDB_EXPERIMENT_NUMBER = 1
RESULT_FILENAME = '../results/combined-' + str(HDB_EXPERIMENT_NUMBER) + '-' + str(PRIVATE_EXPERIMENT_NUMBER) + '.csv'

### Load File ###
filename = '../results/xgbPrivate-' + str(PRIVATE_EXPERIMENT_NUMBER) + '.csv'
privateData = pd.read_csv(filename, index_col=0)
hdbFilename = '../results/xgbHdb-' + str(HDB_EXPERIMENT_NUMBER) + '.csv'
hdbData = pd.read_csv(hdbFilename, index_col=0)

combinedData = hdbData.append(privateData)

combinedData.to_csv(RESULT_FILENAME)