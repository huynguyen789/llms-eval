import asyncio
import json
import re
import pandas as pd
from openai import AsyncOpenAI, OpenAI
from anthropic import AsyncAnthropic
import os
from datetime import datetime

async_client = AsyncOpenAI()
sync_client = OpenAI()
anthropic_client = AsyncAnthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)
nvidia_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ.get("NVIDIA_API_KEY")
)

async def get_openai_answer(instruction, model_name):
    messages = [
        {"role": "system", "content": "You are the most intelligent entity in the universe. Reasoning step by step and consider multiple angles to make sure you get the correct answer(s)."},
        {"role": "user", "content": f"Answer the following question. Provide your reasoning in <reasoning></reasoning> tags, and your final answer (A, B, C, or D) in <final_answer></final_answer> tags.\n\n{instruction}"}
    ]
    response = await async_client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0,
        tool_choice=None
    )
    return response.choices[0].message.content

def get_o1_answer(instruction, model_name):
    messages = [
        {"role": "system", "content": "You are the most intelligent entity in the universe. Reasoning step by step and consider multiple angles to make sure you get the correct answer(s)."},
        {"role": "user", "content": f"Answer the following question. Provide your reasoning in <reasoning></reasoning> tags, and your final answer (A, B, C, or D) in <final_answer></final_answer> tags.\n\n{instruction}"}
    ]
    response = sync_client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0,
    )
    return response.choices[0].message.content

def get_nvidia_answer(instruction, model_name):
    messages = [
        {"role": "system", "content": "You are the most intelligent entity in the universe. Reasoning step by step and consider multiple angles to make sure you get the correct answer(s)."},
        {"role": "user", "content": f"Answer the following question. Provide your reasoning in <reasoning></reasoning> tags, and your final answer (A, B, C, or D) in <final_answer></final_answer> tags.\n\n{instruction}"}
    ]
    response = nvidia_client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0,
        max_tokens=1024
    )
    return response.choices[0].message.content

async def get_anthropic_answer(instruction, model_name):
    message = await anthropic_client.messages.create(
        model=model_name,
        max_tokens=1024,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": f"You are the most intelligent entity in the universe. Answer the following question. Provide your reasoning in <reasoning></reasoning> tags, and your final answer (A, B, C, or D) in <final_answer></final_answer> tags.\n\n{instruction}",
            }
        ],
    )
    return message.content[0].text

def extract_final_answer(model_answer):
    match = re.search(r'<final_answer>(.*?)</final_answer>', model_answer, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        print(f"Warning: Could not extract final answer from model response: {model_answer}")
        return "N/A"

async def process_item(item, model_name):
    if model_name.startswith("claude"):
        if not hasattr(process_item, "anthropic_printed"):
            print(f"Starting to get answers from Anthropic model: {model_name}")
            process_item.anthropic_printed = True
        model_answer = await get_anthropic_answer(item['instruction'], model_name)
    elif model_name.startswith("gpt"):
        if not hasattr(process_item, "openai_printed"):
            print(f"Starting to get answers from OpenAI model: {model_name}")
            process_item.openai_printed = True
        model_answer = await get_openai_answer(item['instruction'], model_name)
    elif model_name.startswith("o1"):
        if not hasattr(process_item, "o1_printed"):
            print(f"Starting to get answers from O1 model: {model_name}")
            process_item.o1_printed = True
        model_answer = get_o1_answer(item['instruction'], model_name)
    elif model_name.startswith("nvidia"):
        if not hasattr(process_item, "nvidia_printed"):
            print(f"Starting to get answers from NVIDIA model: {model_name}")
            process_item.nvidia_printed = True
        model_answer = get_nvidia_answer(item['instruction'], model_name)
    else:
        raise ValueError(f"Unknown model type: {model_name}")
    final_answer = extract_final_answer(model_answer)
    score = 100 if final_answer == item['output'] else 0
    return item['instruction'], item['output'], model_answer, final_answer, score

async def evaluate_model(model_name):
    print(f"\nEvaluating model: {model_name}")
    # Reset the printed flags for each new model evaluation
    process_item.anthropic_printed = False
    process_item.openai_printed = False
    process_item.o1_printed = False
    process_item.nvidia_printed = False
    tasks = [process_item(item, model_name) for item in eval_data]
    results = await asyncio.gather(*tasks)

    df = pd.DataFrame(results, columns=['Instruction', 'Expected Output', 'Model Answer', 'Final Answer', 'Score'])
    avg_score = df['Score'].mean()
    
    print(f"\nModel: {model_name}")
    print(f"Average Score: {avg_score:.2f}")

    excel_path = f'{output_folder}/{dataset_name}_{current_time}_{model_name}.xlsx'
    df.to_excel(excel_path, index=False)
    print(f"Results saved to {excel_path}")

    return df, avg_score

async def main():
    models_to_evaluate = [
        "nvidia/llama-3.1-nemotron-70b-instruct",
        "o1-preview",
        "claude-3-5-sonnet-20240620",
        "gpt-4o-mini", 
        "gpt-4o",  
        "gpt-4-0125-preview"
    ]
    results = {}

    for model in models_to_evaluate:
        df, avg_score = await evaluate_model(model)
        results[model] = {"df": df, "avg_score": avg_score}

    # Create a summary DataFrame
    summary_data = [(model, data["avg_score"]) for model, data in results.items()]
    summary_df = pd.DataFrame(summary_data, columns=["Model", "Average Score"])
    summary_df = summary_df.sort_values("Average Score", ascending=False).reset_index(drop=True)

    print("\nModel Comparison Summary:")
    print(summary_df)

    # Save summary to Excel in the output folder
    summary_excel_path = f'{output_folder}/model_comparison_summary_{current_time}.xlsx'
    summary_df.to_excel(summary_excel_path, index=False)
    print(f"\nSummary saved to {summary_excel_path}")

    return results, summary_df

# Load the evaluation dataset
input_file_path = './input/huy_dataset/huy_test.json'
with open(input_file_path, 'r') as f:
    eval_data = json.load(f)

# Extract dataset name from input file path
dataset_name = os.path.splitext(os.path.basename(input_file_path))[0]

# Create output folder name
current_time = datetime.now().strftime("%m%d%y_%I%M%p")
output_folder = f'./output/{dataset_name}_{current_time}'

# Create the output folder
os.makedirs(output_folder, exist_ok=True)

# Check if we're in a Jupyter notebook
try:
    get_ipython()
    is_notebook = True
except NameError:
    is_notebook = False

if __name__ == "__main__":
    if is_notebook:
        results, summary_df = await main()
    else:
        results, summary_df = asyncio.run(main())
