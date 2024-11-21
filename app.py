import asyncio
import json
import re
import pandas as pd
from openai import AsyncOpenAI, OpenAI
from anthropic import AsyncAnthropic
import os
from datetime import datetime

client = AsyncOpenAI()
anthropic_client = AsyncAnthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)

google_client = OpenAI(
    api_key=os.environ.get("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

async def get_openai_answer(instruction, model_name):
    messages = [
        {"role": "system", "content": "You are the most intelligent entity in the universe. Reasoning step by step and consider multiple angles to make sure you get the correct answer(s)."},
        {"role": "user", "content": instruction}
    ]
    response = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0,
        tool_choice=None
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
              "content": f"You are the most intelligent entity in the universe. Reasoning step by step and consider multiple angles to make sure you get the correct answer(s). Here's the task: {instruction}",
          }
      ],
  )
  return message.content[0].text

def get_google_answer(instruction, model_name):
    """
    Input: instruction (str), model_name (str)
    Process: Makes synchronous call to Google's API
    Output: Returns model's response text
    """
    response = google_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are the most intelligent entity in the universe. Reasoning step by step and consider multiple angles to make sure you get the correct answer(s)."},
            {"role": "user", "content": instruction}
        ],
        temperature=0.0
    )
    return response.choices[0].message.content

async def evaluate_answer(model_answer, expected_output):
  messages = [
      {"role": "system", "content": "You are a world-class AI model evaluator. "},
      {"role": "user", "content": f"""
       Model answer: {model_answer}\n\n
       Expected output: {expected_output}\n\n
       Your task is to compare the model's answer WITH THE EXPECTED OUTPUT and provide a super concise reason in one short sentence for the score, and then a score from 0 to 100. 
       Example: Reason: [super concise reason here]. Score: [score here]. 
       Use the following scale: 0 is completely wrong, 50 is missing half of the solution, 100 is completely correct, 80-90 if correct but missing some detail or not a complete answer. 
       Don't grade on formatting, as long as the answer is correct compare to the expected output. 
       If the logic is correct but the final answer is wrong, it's still wrong.
       If the answer is correct but it has extra information, it's still correct. As long as the extra info is not completely wrong or hallucinated.
       Do not grade by your knowledge, but grade based on the expected output. 
       Always include the numeric score (0-100) in your response.
       """}
  ]
  response = await client.chat.completions.create(
      model="gpt-4o",
      messages=messages,
      temperature=0.0,
      tool_choice=None
  )
  return response.choices[0].message.content

def extract_score_and_reason(evaluation):
  match = re.search(r'Reason:\s*(.*?)\s*Score:\s*(\d+|10)', evaluation, re.IGNORECASE | re.DOTALL)
  if match:
      reason = match.group(1).strip()
      score = int(match.group(2))
      return score, reason
  else:
      print(f"Warning: Could not extract score and reason from evaluation: {evaluation}")
      return 0, "Unable to extract reason"  # Default values if extraction fails

async def process_item(item, model_name):
    """
    Handle different model types (OpenAI, Anthropic, Google)
    """
    if model_name.startswith("claude"):
        model_answer = await get_anthropic_answer(item['instruction'], model_name)
    elif model_name.startswith("gemini"):
        # Call sync function for Google models
        model_answer = get_google_answer(item['instruction'], model_name)
    else:
        model_answer = await get_openai_answer(item['instruction'], model_name)
    
    evaluation = await evaluate_answer(model_answer, item['output'])
    score, reason = extract_score_and_reason(evaluation)
    return item['instruction'], item['output'], model_answer, score, reason

async def evaluate_model(model_name):
    tasks = [process_item(item, model_name) for item in eval_data]
    results = await asyncio.gather(*tasks)

    df = pd.DataFrame(results, columns=['Instruction', 'Expected Output', 'Model Answer', 'Score', 'Reason'])
    avg_score = df['Score'].mean()
    
    print(f"\nModel: {model_name}")
    print(f"Average Evaluation Score: {avg_score:.2f}")

    excel_path = f'{output_folder}/{dataset_name}_{current_time}_{model_name}.xlsx'
    df.to_excel(excel_path, index=False)
    print(f"Results saved to {excel_path}")

    return df, avg_score

async def main():
  models_to_evaluate = [
    #   "claude-3-5-sonnet-20241022",
    #     "gpt-4o-2024-11-20",
    #   "gpt-4o-mini", 
    #   "gpt-4o", 
    #   "gpt-4-0125-preview",
      "gemini-1.5-pro",  # Add Google models
      "gemini-1.5-flash"
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
  summary_excel_path = f'{output_folder}/model_comparison_summary.xlsx'
  summary_df.to_excel(summary_excel_path, index=False)
  print(f"\nSummary saved to {summary_excel_path}")

  return results, summary_df

# Load the evaluation dataset
input_file_path = './input/huy_dataset/huy_test.json'
# './input/huy_dataset/huy_test.json'
# 'input/math-EvalDataset-10.json'
#'./input/huy_dataset/huy_test2.json'
# './input/EvalDataset-20.json'


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

if is_notebook:
    results, summary_df = await main()
else:
    results, summary_df = asyncio.run(main())