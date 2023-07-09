#!/bin/bash

# Task A: Scoring transcripts
output_types=("numerical" "numerical_reasoning" "numerical_descriptions")

for output_type in "${output_types[@]}"; do
    echo "CLASS on $output_type"
    python3 scripts/class_obs/compare.py --rating_output=$output_type

    echo "MQI on $output_type"
    python3 scripts/mqi_obs/compare.py --rating_output=$output_type
done


# Task B: Identify highlights and missed opportunities
python3 scripts/plot_task_bc_results.py --key=class_examples
python3 scripts/plot_task_bc_results.py --key=mqi_examples

# Task C: Provide suggestions
python3 scripts/plot_task_bc_results.py --key=suggestions