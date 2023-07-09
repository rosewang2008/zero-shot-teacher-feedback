import os
DATA_DIR = 'data'
OUTCOME_FILE = 'class_data.csv'
OUTCOME_KEYS = [
    'CLPC', # positive climate, dimension: emotinal support
    'CLBM', # behavior management, dimension: classroom organization
    'CLINSTD', # instructional dialogue, dimension: instructional support
    ]
OUTCOMEKEY2LABELNAME = {
    'CLPC': 'Positive Climate',
    'CLBM': 'Behavior Management',
    'CLINSTD': 'Instructional Dialogue',
}
MODEL_NAME = 'gpt-3.5-turbo'
RESULTS_DIR = os.path.join('results', 'class')