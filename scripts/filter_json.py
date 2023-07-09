import json 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_fname', type=str, required=True)
parser.add_argument('--output_fname', type=str, required=True)


VALID_IDS = [
    '24e4b45f-37ae-4271-8c62-e51f94502c37',
    'bfed608f-8026-435d-bc07-3a064c77ef1e',
    '4bb6f876-bb17-4191-bbd9-30d844c463eb',
    '7a206efe-971c-4cff-a825-a79ff291aa39',
    '7f4104bb-1381-4a35-a093-aa78cc85f5f8',
    'ac915fe2-772b-47a2-a841-1d0e25e90edd',
]

TASK2NUM_RESPONSES = {
    'suggestions': 21,
    'mqi_examples': 161,
    'class_examples': 121
}

def load_json(fname):
    with open(fname, 'r') as f:
        data = json.load(f)
    return data

def filter_by_ids(data, valid_ids=VALID_IDS):
    result =  [x for x in data if x['id'] in valid_ids]
    return result

def filter_by_done(data, task2num_responses=TASK2NUM_RESPONSES):
    return [x for x in data if len(x['responses']) >= task2num_responses[x['task']]]

def filter_by_task(data, tasks=['suggestions', 'mqi_examples', 'class_examples']):
    return [x for x in data if x['task'] in tasks]