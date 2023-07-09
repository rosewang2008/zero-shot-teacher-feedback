"""
python3 plot_results.py --key suggestions

python3 plot_results.py --key mqi_examples

python3 plot_results.py --key class_examples

"""
import sys
import os
sys.path.append(os.getcwd())

from scripts.filter_json import (
    load_json,
    filter_by_task,
)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import argparse
import numpy as np
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
)

parser = argparse.ArgumentParser()
parser.add_argument('--key', type=str, default='suggestions')
args = parser.parse_args()


PLOT_DIR = f'results/{args.key}'

# Mkdir if it doesn't exist
import os
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

KEY2TITLE = {
    'suggestions': 'Human evaluations of model suggestions',
    'mqi_examples': 'Human evaluations of model MQI examples',
    'class_examples': 'Human evaluations of model CLASS examples',
}

RATING2INT = {
    'low': 0,
    'mid': 1,
    'high': 2,
    '': None
}

if args.key == 'suggestions':
    KEYS = ['faithful', 'relevant', 'actionable', 'novel']
    TARGET_NUM_KEYS = 4
else:
    KEYS = ['faithful', 'relevant', 'insightful']
    TARGET_NUM_KEYS = 3

# Colors for low, mid, high
COLORS = ['#EE8866', '#EEDD88', '#44BB99'] # color blind friendly 

def plot_stacked_plot(df, output_fname, title):
    # Stacked bar plot of the ratings for each key: eg. relevant: [low, mid, high]
    # Count occurrences of each rating per key
    rating_counts = df.groupby(['key', 'nl_rating']).rating.size().reset_index(name='count')
    # Count total occurrences of each key
    total_counts = df.groupby('key').size().reset_index(name='total_count')
    # Merge the two dfs 
    merged_df = pd.merge(rating_counts, total_counts, on='key')
    # Calculate the proportion of each rating per key
    merged_df['proportion'] = merged_df['count'] / merged_df['total_count']

    # # Print proportion of each rating per key
    # print(merged_df)

    # Pretty plot
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
    bottom = np.zeros(len(KEYS))
    for idx, rating in enumerate(['low', 'mid', 'high']):
        # Get the proportion of each rating per key
        data = merged_df[merged_df.nl_rating == rating]
        
        # If one of the ratings is missing, then we need to add it in
        if len(data) != TARGET_NUM_KEYS:
            for key in KEYS:
                if key not in data.key.values:
                    data = data.append({
                        'key': key, 
                        'nl_rating': rating, 
                        'count': 0, 
                        'total_count': total_counts[total_counts['key'] == key]['total_count'].values[0], 
                        'proportion': 0
                    }, ignore_index=True)
            data = data.sort_values(by='key')
        
        # Plot in order of KEYS
        sns.barplot(x='key', y='proportion', data=data, color=COLORS[idx], bottom=bottom, order=KEYS)


        # Add text showing the proportion, center the text in the middle of the bar
        for key, v in zip(data.key, data.proportion):
            i = KEYS.index(key)
            plt.text(i, bottom[i] + v/2, f"{v*100:.0f}%", ha='center', va='center', color='black', fontsize=14)

        # Set bottom to the previous bottom + the current proportion according to KEYS order
        bottom = [bottom[KEYS.index(key)] + data[data.key == key]['proportion'].values[0] for key in KEYS]


    # Customize the plot
    # plt.xlabel('Key', fontsize=14)
    # Remove x label
    plt.xlabel('')
    plt.ylabel('Proportion (%)', fontsize=14)
    plt.title(title, fontsize=16)

    # Plot legend outside of plot
    plt.legend(
        labels=['No', 'Somewhat', 'Yes'],
        # Show color of legend
        handles=[
            plt.Rectangle((0,0),1,1, color=COLORS[0], ec="k"),
            plt.Rectangle((0,0),1,1, color=COLORS[1], ec="k"),
            plt.Rectangle((0,0),1,1, color=COLORS[2], ec="k"),
        ],
        loc='upper center',
        bbox_to_anchor=(0.5, -0.07),
        fancybox=True,
        # shadow=True,
        ncol=3
    )

    # # Save figure
    # plt.savefig(f'{PLOT_DIR}/{output_fname}.png', bbox_inches='tight')
    # Save as pdf
    plt.savefig(f'{PLOT_DIR}/{output_fname}.pdf', bbox_inches='tight')

    # # Show the plot
    # plt.show()

def calculate_interannotator_agreement(df):
    # Calculate the inter-annotator agreement: get name == 'rater1' and name == 'rater2'  on the same unique_id
    rater1 = df[df['name'] == 'rater1']
    rater2 = df[df['name'] == 'rater2']
    # Drop nan in rating
    rater1 = rater1.dropna(subset=['rating'])
    rater2 = rater2.dropna(subset=['rating'])

    # TARGET_NUM = 3 for examples, 4 for suggestions
    TARGET_NUM = 3 if 'examples' in args.key else 4
    # Only keep the intersection that has 3 entries in rater1 and rater2
    rater1 = rater1.groupby('unique_id').filter(lambda x: len(x) == TARGET_NUM)
    rater2 = rater2.groupby('unique_id').filter(lambda x: len(x) == TARGET_NUM)

    # Find the intersection of unique_ids
    intersection = set(rater1.unique_id).intersection(set(rater2.unique_id))
    # Filter by intersection and sort by unique_id
    rater1 = rater1[rater1.unique_id.isin(intersection)].sort_values(by=['unique_id'])
    rater2 = rater2[rater2.unique_id.isin(intersection)].sort_values(by=['unique_id'])

    # Check that the unique_ids are the same
    assert len(rater1.unique_id.values) == len(rater2.unique_id.values)
    same = all([rater1.unique_id.values[i] == rater2.unique_id.values[i] for i in range(len(rater1.unique_id.values))])
    assert same

    labels = [0,1,2]

    # print(f'Number of samples: {len(rater1)}')

    # Calculate the agreement
    agreement = cohen_kappa_score(rater1['rating'], rater2['rating'], labels=labels)
    # print(f'Agreement between rater1 and rater2: {agreement} on {args.key}')

    # Split by key
    for key in df.key.unique():
        r1_key = rater1[rater1['key'] == key]
        r2_key = rater2[rater2['key'] == key]
        agreement = cohen_kappa_score(r1_key['rating'], r2_key['rating'], labels=labels)
        # print(f'Agreement between rater1 and rater2 on {key}: {agreement} || num samples {len(r1_key)}')

        # Also try quadratic weighted kappa
        agreement = cohen_kappa_score(r1_key['rating'], r2_key['rating'], labels=labels, weights='quadratic')
        # print(f'Quadratic agreement between rater1 and rater2 on {key}: {agreement} || num samples {len(r1_key)}')
        
        # print(f"rater1's rating: {r1_key['rating'].values}")
        # print(f"rater2's rating: {r2_key['rating'].values}")



    # Create confusion matrix for each key, rater1 on x-axis, rater2 on y-axis
    for key in df.key.unique():
        r1_key = rater1[rater1['key'] == key]
        r2_key = rater2[rater2['key'] == key]
        # Create confusion matrix
        cm = confusion_matrix(r1_key['rating'], r2_key['rating'], labels=labels)
        # print(f'Confusion matrix for {key}:')
        # print(cm)
        # Show percentage of each cell
        cm_perc = cm / cm.sum()
        # Round to 2 decimal places
        cm_perc = np.round(cm_perc, 2)
        # print(f'Confusion matrix percentage for {key}:')
        # print(cm_perc)

        # Plot confusion matrix
        sns.heatmap(cm_perc, annot=True, fmt='g', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Rater 1')
        plt.ylabel('Rater 2')
        # Make x axis ticks show up on top
        plt.tick_params(axis='x', which='both', bottom=False, top=True, labelbottom=False, labeltop=True)
        plt.title(f'Confusion matrix for {key}')
        # plt.savefig(f'{PLOT_DIR}/confusion_matrix_{key}.png', bbox_inches='tight')
        plt.savefig(f'{PLOT_DIR}/confusion_matrix_{key}.pdf', bbox_inches='tight')
        # plt.show()
        plt.clf()
    

if __name__ == '__main__':
    results = load_json('data/task-bc-annotations.json')

    if args.key == 'examples':
        results = filter_by_task(results, ['class_examples', 'mqi_examples'])
    else:
        results = filter_by_task(results, [args.key])

    
    df = []
    for session in tqdm.tqdm(results):
        task = session['task']
        name = session['name']
        # Print session_transcript
        # Then 'model_response'
        # Then 'relevant' -> 'relevant_comment'
        # same with faithful, useful, redundant
        segment_transcripts = session['segment_transcripts']
        responses = session['responses'][1:] # Skip the first one because that's the practice
        model_prompts = session['model_prompts']
        assert len(segment_transcripts) == len(responses)

        for i, (segment_transcript, response) in enumerate(zip(segment_transcripts, responses)):
            NCTETID = response['NCTETID']
            OBSID = response['OBSID']
            CHAPNUM = response['CHAPNUM']
            model_prompt_key = response['model_prompt_key']
            model_response_id = response['model_response_id']
            model_prompt = model_prompts[i]
            model_response = response['model_response']
            if segment_transcript is None:
                print('Skipping segment because it is None')
                continue
        
            for key in ['relevant', 'faithful', 'useful', 'redundant']:
                rating = RATING2INT[response[key]]
                result = {
                    'name': name,
                    'task': task,
                    'segment': i,
                    'key': key,
                    'nl_rating': response[key],
                    'rating': rating,
                    'model_response': model_response,
                    'unique_id': f'{NCTETID}_{OBSID}_{CHAPNUM}_{model_prompt_key}_{model_response_id}',
                }
                df.append(result)
    
    df = pd.DataFrame(df)
    

    if args.key == 'suggestions':
        # Rename key='redundant' to 'novel' 
        df['key'] = df['key'].replace('redundant', 'novel')
        # Only flip the nl_rating for key='novel': low -> high, high -> low
        df.loc[df.key == 'novel', 'nl_rating'] = df.loc[df.key == 'novel', 'nl_rating'].map({'low': 'high', 'high': 'low', 'mid': 'mid'})
        df['key'] = df['key'].replace('useful', 'actionable') 
    else: 
        # Rename useful to 'insightful'
        df['key'] = df['key'].replace('useful', 'insightful')


    # Plot stacked bar plot
    plot_stacked_plot(
        df, 
        output_fname='stacked_bar_plot', 
        title=KEY2TITLE[args.key])

    calculate_interannotator_agreement(df)