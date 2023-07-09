DATA_DIR = 'data'
OUTCOME_FILE = 'mqi_data.csv'
OUTCOME_KEYS = [
        'EXPL', # explanation, dimension: richness of the mathematical language
        'REMED', # remediation, dimension: Working with Students and Mathematics
        'LANGIMP', # language imprecision, dimension: Errors and Imprecision
        'SMQR', # student mathematical questioning and reasoning, dimension: Student Participation
    ]
OUTCOMEKEY2LABELNAME = {
    'EXPL': 'Explanations',
    'REMED': 'Remediation of Student Errors and Difficulties',
    'LANGIMP': 'Imprecision in Language or Notation',
    'SMQR': 'Student Mathematical Questioning and Reasoning',
}
RESULTS_DIR = 'results/mqi' 
MODEL_NAME = 'gpt-3.5-turbo'