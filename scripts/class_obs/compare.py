import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
sys.path.append(os.getcwd())

import seaborn as sns
import argparse

from scripts.class_obs.constants import (
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


MODEL_OUTCOME_KEYS = [f"{MODEL_NAME}_{k}_{args.rating_output}" for k in OUTCOME_KEYS]
MODEL_PROMPT_KEYS = ['{}_prompt'.format(k) for k in MODEL_OUTCOME_KEYS]

PLOT_DIR = os.path.join(RESULTS_DIR, args.rating_output)

# Make results dir if it doesn't exist
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)


def plot_stacked_barplot(results_df): 
    # Plot stacked barplot per outcome
    for model_k, model_prompt_k, k in zip(MODEL_OUTCOME_KEYS, MODEL_PROMPT_KEYS, OUTCOME_KEYS):
        # Only plot the obsids that have a value for this outcome
        plot_df = results_df[~results_df[model_k].isna()]
        # And model_prompt_k is not null
        plot_df = plot_df[~plot_df[model_prompt_k].isna()]
        # Convert model_k to int
        plot_df[model_k] = plot_df[model_k].astype(float)

        # Pretty plot 
        sns.set_style("whitegrid")
        sns.set_context("paper")
        sns.set(font_scale=1.2)

        # Plot stacked barplot
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
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
                'percentage': percentage
            })
        for rating in unique_model:
            # Get the percentage of model ratings
            model = plot_df[plot_df[model_k] == rating]
            percentage = len(model) / len(plot_df)
            percentages_df.append({
                'rating': rating,
                'rater': 'model',
                'percentage': percentage
            })
        # Convert to dataframe
        percentages = pd.DataFrame(percentages_df)
        # Plot the barplot - put human and model side by side
        sns.barplot(
            x='rating', y='percentage', hue='rater', data=percentages, ax=ax,
                    # color for human and model
                    palette=['#BBCC33', '#99DDFF'])
        # Label the percentages
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2., height + 0.01, '{:1.0f}%'.format(100*height), ha="center")

        # Y axis 0-1
        ax.set_ylim(0, 1)
        plot_fname = os.path.join(PLOT_DIR, '{}_stacked_barplot.pdf'.format(model_k))
        # Title is outcome
        ax.set_title(f"{OUTCOMEKEY2LABELNAME[k]} ({k})")
        # X lab
        ax.set_xlabel('CLASS rating (1-7)')
        # Y lab
        ax.set_ylabel('Percentage of ratings')
        # Legend 
        ax.legend()

        plt.savefig(plot_fname)
        print("Saved plot to {}".format(plot_fname))
        plt.close()

def plot_scatterplot(results_df):
    # x = CLPC, y = CLPC_gpt-3.5-turbo
    for model_k, model_prompt_k, k in zip(MODEL_OUTCOME_KEYS, MODEL_PROMPT_KEYS, OUTCOME_KEYS):
        # Only plot the obsids that have a value for this outcome
        plot_df = results_df[~results_df[model_k].isna()]
        # And model_prompt_k is not null
        plot_df = plot_df[~plot_df[model_prompt_k].isna()]
        # Convert model_k to int
        plot_df[model_k] = plot_df[model_k].astype(float)
        # Plot k and model_k side by side
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        sns.scatterplot(data=plot_df, x=k, y=model_k, ax=ax)
        # Fit line
        sns.regplot(data=plot_df, x=k, y=model_k, ax=ax, scatter=False)
        
        # Set x axis to be the same 0, 7
        ax.set_xlim(1, 7)
        ax.set_ylim(1, 7)

        # Title is the correlation
        corr = plot_df[[k, model_k]].corr().iloc[0, 1]
        ax.set_title("Correlation: {:.2f}".format(corr))

        # Print spearman correlation
        spearman_corr = plot_df[[k, model_k]].corr(method='spearman').iloc[0, 1]
        print()
        print(f"Spearman correlation: {spearman_corr} on {model_k}")
        
        # Save plot
        # plot_fname = os.path.join(PLOT_DIR, '{}_scatterplot.png'.format(model_k))
        # plt.savefig(plot_fname)
        plot_fname = os.path.join(PLOT_DIR, '{}_scatterplot.pdf'.format(model_k))
        plt.savefig(plot_fname)
        print("Saved plot to {}".format(plot_fname))
        plt.close()


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
    CODE1 = 'CLPC'
    CODE2 = 'CLBM'
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
    ax[0].set_xticklabels(ax[0].get_xticklabels(), ha='center')
    ax[1].set_xticklabels(ax[1].get_xticklabels(), ha='center')

    # Set y labels
    ax[0].set_ylabel('Percentage of ratings')
    ax[0].set_ylim(0, 1)
    ax[1].set_ylim(0, 1)

    ax[0].set_title(f"{OUTCOMEKEY2LABELNAME[CODE1]} ({CODE1})", pad=8)
    ax[1].set_title(f"{OUTCOMEKEY2LABELNAME[CODE2]} \n ({CODE2})", pad=8)

    # Remove x label
    ax[0].set_xlabel('')
    ax[1].set_xlabel('')

    # Remove y label on second plot
    ax[1].set_ylabel('')

    # Only have one legend 
    ax[1].legend_.remove()

    plot_fname = os.path.join(PLOT_DIR, f'{CODE1}_{CODE2}_histogram.pdf')
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


    