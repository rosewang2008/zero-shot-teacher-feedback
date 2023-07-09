import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
sys.path.append(os.getcwd())
import seaborn as sns
import argparse 

from scripts.mqi_obs.constants import (
    OUTCOME_KEYS,
    DATA_DIR,
    OUTCOME_FILE,
    MODEL_NAME, 
    RESULTS_DIR,
    OUTCOMEKEY2LABELNAME
)

parser = argparse.ArgumentParser()
parser.add_argument('--rating_output', type=str, default='numerical')

args = parser.parse_args()

MAX_SCORE = 3

MODEL_OUTCOME_KEYS = ['{}_{}_{}'.format(MODEL_NAME, k, args.rating_output) for k in OUTCOME_KEYS]
MODEL_PROMPT_KEYS = ['{}_prompt'.format(k) for k in MODEL_OUTCOME_KEYS]


PLOT_DIR = os.path.join(RESULTS_DIR, args.rating_output)
print('PLOT_DIR', PLOT_DIR)

# Make results dir if it doesn't exist
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)


def plot_specific_histogram(results_df):
    percentages_df = []
    # Plot histogram per outcome
    for model_k, model_prompt_k, k in zip(MODEL_OUTCOME_KEYS, MODEL_PROMPT_KEYS, OUTCOME_KEYS):
        # Only plot the obsids that have a value for this outcome
        plot_df = results_df[~results_df[model_k].isna()]
        # And where prompt is not null
        plot_df = plot_df[~plot_df[model_prompt_k].isna()]

        # Remove rows where if results[model_k] is a string, then remove if > 5
        for i, row in plot_df.iterrows():
            if isinstance(row[model_k], str) and len(row[model_k].split()) > 5:
                plot_df = plot_df.drop(i)
            # check if == 'na' or 'n/a' or 'nan' or 'None'
            elif isinstance(row[model_k], str) and row[model_k].lower() in ['na', 'n/a', 'nan', 'none']:
                plot_df = plot_df.drop(i)
            else: # change str -> int, float -> int
                plot_df.at[i, model_k] = int(float(row[model_k]))
        # Concert model_k to float
        plot_df[model_k] = plot_df[model_k].astype(float)

        # Pretty plot 
        sns.set_style("whitegrid")
        sns.set_context("paper")
        sns.set(font_scale=1.7)

        # Plot stacked barplot
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        # FOr every unique rating, calculate the percentage of human and model ratings
        unique_human = plot_df[k].unique()
        unique_model = plot_df[model_k].unique()
        for rating in unique_human:
            # Get the percentage of human ratings
            human = plot_df[plot_df[k] == rating]
            percentage = len(human) / len(plot_df)
            percentages_df.append({
                'rating': rating,
                'rater': 'human',
                'percentage': percentage,
                'code': k,
            })
        for rating in unique_model:
            # Get the percentage of model ratings
            model = plot_df[plot_df[model_k] == rating]
            percentage = len(model) / len(plot_df)
            percentages_df.append({
                'rating': rating,
                'rater': 'model',
                'percentage': percentage,
                'code': k,
            })

    # Convert to dataframe
    percentages = pd.DataFrame(percentages_df)

    # Plot two barplots side by side - made it wider
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))

    # plot code = EXPL 
    CODE1 = 'EXPL'
    CODE2 = 'REMED'
    # Plot the barplot - put human and model side by side
    sns.barplot(
        x='rating', 
        y='percentage', 
        hue='rater', 
        data=percentages[percentages['code'] == CODE1],
        ax=ax[0],
        # color for human and model
        palette=['#BBCC33', '#99DDFF']
        )
    # Label the percentages
    for i, p in enumerate(ax[0].patches):
        height = p.get_height()
        # Change fontsize
        ax[0].text(p.get_x() + p.get_width()/2., height + 0.01, '{:1.0f}%'.format(100*height), ha="center", fontsize=15)
    
    # plot code = REMED
    # Plot the barplot - put human and model side by side
    sns.barplot(
        x='rating',
        y='percentage',
        hue='rater',
        data=percentages[percentages['code'] == CODE2],
        ax=ax[1],
        # color for human and model
        palette=['#BBCC33', '#99DDFF']
        )
    # Label the percentages
    for i, p in enumerate(ax[1].patches):
        height = p.get_height()
        ax[1].text(p.get_x() + p.get_width()/2., height + 0.01, '{:1.0f}%'.format(100*height), ha="center", fontsize=15)

    # Set x label to be in the middle
    ax[0].set_xticklabels(['1', '2', '3'])
    ax[1].set_xticklabels(['1', '2', '3'])

    # Set y labels
    ax[0].set_ylabel('Percentage of ratings')
    ax[0].set_ylim(0, 1)
    ax[1].set_ylim(0, 1)

    ax[0].set_title(f"{OUTCOMEKEY2LABELNAME[CODE1]} ({CODE1})", pad=8)
    # ax[1].set_title(f"{OUTCOMEKEY2LABELNAME[CODE2]} ({CODE2})")
    # Split title into two lines evenly
    split_title = f"{OUTCOMEKEY2LABELNAME[CODE2]} ({CODE2})".split()
    split_title = ' '.join(split_title[:len(split_title)//2]) + '\n' + ' '.join(split_title[len(split_title)//2:])
    ax[1].set_title(split_title, pad=8)
    # ax[1].set_title(f"{OUTCOMEKEY2LABELNAME[CODE2]} \n ({CODE2})")

    # Remove x label
    ax[0].set_xlabel('')
    ax[1].set_xlabel('')

    # Remove y label on second plot
    ax[1].set_ylabel('')

    # Only have one legend 
    ax[0].legend_.remove()
    ax[1].legend_.remove()

    # X lab
    # ax.set_xlabel('MQI rating (1-3)')
    # Y lab
    # Title is outcome
    # plot_fname = os.path.join(PLOT_DIR, '{}_histogram.png'.format(model_k))
    # plt.savefig(plot_fname)
    plot_fname = os.path.join(PLOT_DIR, f'{CODE1}_{CODE2}_histogram.pdf')
    plt.savefig(plot_fname)
    print("Saved plot to {}".format(plot_fname))
    plt.close()


def plot_stacked_barplot(results_df):
    # Plot histogram per outcome
    for model_k, model_prompt_k, k in zip(MODEL_OUTCOME_KEYS, MODEL_PROMPT_KEYS, OUTCOME_KEYS):
        # Only plot the obsids that have a value for this outcome
        plot_df = results_df[~results_df[model_k].isna()]
        # And where prompt is not null
        plot_df = plot_df[~plot_df[model_prompt_k].isna()]

        # Remove rows where if results[model_k] is a string, then remove if > 5
        for i, row in plot_df.iterrows():
            if isinstance(row[model_k], str) and len(row[model_k].split()) > 5:
                plot_df = plot_df.drop(i)
            # check if == 'na' or 'n/a' or 'nan' or 'None'
            elif isinstance(row[model_k], str) and row[model_k].lower() in ['na', 'n/a', 'nan', 'none']:
                plot_df = plot_df.drop(i)
            else: # change str -> int, float -> int
                plot_df.at[i, model_k] = int(float(row[model_k]))
        # Concert model_k to float
        plot_df[model_k] = plot_df[model_k].astype(float)

        # Pretty plot 
        sns.set_style("whitegrid")
        sns.set_context("paper")
        sns.set(font_scale=1.7)

        # Plot stacked barplot
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        # FOr every unique rating, calculate the percentage of human and model ratings
        unique_human = plot_df[k].unique()
        unique_model = plot_df[model_k].unique()
        percentages_df = []
        for rating in unique_human:
            # Get the percentage of human ratings
            human = plot_df[plot_df[k] == rating]
            percentage = len(human) / len(plot_df)
            percentages_df.append({
                'rating': rating,
                'rater': 'human',
                'percentage': percentage,
                'code': k,
            })
        for rating in unique_model:
            # Get the percentage of model ratings
            model = plot_df[plot_df[model_k] == rating]
            percentage = len(model) / len(plot_df)
            percentages_df.append({
                'rating': rating,
                'rater': 'model',
                'percentage': percentage,
                'code': k,
            })
        # Convert to dataframe
        percentages = pd.DataFrame(percentages_df)
        # Plot the barplot - put human and model side by side
        sns.barplot(x='rating', y='percentage', hue='rater', data=percentages, ax=ax, 
                    # color for human and model
                    palette=['#BBCC33', '#99DDFF'])
        # Label the percentages
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2., height + 0.01, '{:1.0f}%'.format(100*height), ha="center")


        # X lab
        ax.set_xlabel('MQI rating (1-3)')
        # Y lab
        ax.set_ylabel('Percentage of ratings')
        ax.set_ylim(0, 1)
        # Title is outcome
        ax.set_title(f"{OUTCOMEKEY2LABELNAME[k]} ({k})")
        # plot_fname = os.path.join(PLOT_DIR, '{}_histogram.png'.format(model_k))
        # plt.savefig(plot_fname)
        plot_fname = os.path.join(PLOT_DIR, '{}_stacked_barplot.pdf'.format(model_k))
        plt.savefig(plot_fname)
        print("Saved plot to {}".format(plot_fname))
        plt.close()

def plot_scatterplot(results_df):
    # x = CLPC, y = CLPC_gpt-3.5-turbo
    for model_k, model_prompt_k, k in zip(MODEL_OUTCOME_KEYS, MODEL_PROMPT_KEYS, OUTCOME_KEYS):
        avg_k = 'avg_{}'.format(k)
        # Only plot the obsids that have a value for this outcome
        plot_df = results_df[~results_df[model_k].isna()]
        # And where prompt is not null
        plot_df = plot_df[~plot_df[model_prompt_k].isna()]
        # Remove rows where if results[model_k] is a string, then remove if > 5
        for i, row in plot_df.iterrows():
            if isinstance(row[model_k], str) and len(row[model_k].split()) > 5:
                plot_df = plot_df.drop(i)
            # check if == 'na' or 'n/a' or 'nan' or 'None'
            elif isinstance(row[model_k], str) and row[model_k].lower() in ['na', 'n/a', 'nan', 'none']:
                plot_df = plot_df.drop(i)
            else: # change str -> int, float -> int
                plot_df.at[i, model_k] = float(row[model_k])

        # Average the scores for k, within every OBSID and CHAPNUM
        plot_df[avg_k] = plot_df.groupby(['OBSID', 'CHAPNUM'])[k].transform('mean')
            
        
        # Set type to be float
        plot_df[avg_k] = plot_df[avg_k].astype(float)
        plot_df[model_k] = plot_df[model_k].astype(float)
        
        # Plot k and model_k side by side
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        sns.scatterplot(data=plot_df, x=avg_k, y=model_k, ax=ax)
        # Fit line
        sns.regplot(data=plot_df, x=avg_k, y=model_k, ax=ax, scatter=False)
        # Set x axis to be the same 0, MAX_SCORE
        ax.set_xlim(1, MAX_SCORE)
        ax.set_ylim(1, MAX_SCORE)

        # Title is the correlation
        corr = plot_df[[avg_k, model_k]].corr().iloc[0, 1]
        ax.set_title("Correlation: {:.2f}".format(corr))

        # Print spearman correlation
        spearman_corr = plot_df[[k, model_k]].corr(method='spearman').iloc[0, 1]
        print()
        print(f"Spearman correlation: {spearman_corr} on {model_k}")
        
        # # Save plot
        # plot_fname = os.path.join(PLOT_DIR, '{}_avg_scatterplot.png'.format(model_k))
        # plt.savefig(plot_fname)
        plot_fname = os.path.join(PLOT_DIR, '{}_avg_scatterplot.pdf'.format(model_k))
        plt.savefig(plot_fname)
        print("Saved plot to {}".format(plot_fname))
        plt.close()



if __name__ == "__main__":
    # Load data
    outcome_df = pd.read_csv(os.path.join(DATA_DIR, OUTCOME_FILE))
    results_df = pd.read_csv(os.path.join(RESULTS_DIR, 'results_{}.csv'.format(MODEL_NAME)))
    
    plot_scatterplot(results_df)
    plot_stacked_barplot(results_df)
    plot_specific_histogram(results_df)


    