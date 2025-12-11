# Operational Security L4 Benchmark (L2_SEC_OpSec)

This repository implements **Automated Document Analysis (ADA)** for the
Operational Security dimension of an AI Ethics Benchmark.

L2_SEC_OpSec is decomposed into eight Level-4 indicators:

1. `L4_SEC_SecretsManagement`  
2. `L4_SEC_AccountTakeoverDetection`  
3. `L4_SEC_ContinuityRecovery`  
4. `L4_SEC_EndpointScrapingEfficacy`   

Each indicator has:

- A structured evaluation prompt with criteria and scoring rules  
- A JSON schema for model output  
- A Python pipeline that:
  - reads PDFs/DOCX/TXT  
  - chunks long documents  
  - calls a chat model (e.g., `gpt-4o`)  
  - saves per-chunk JSON and a merged result under `outputs/<L4>/<model>/`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then edit and add your OPENAI_API_KEY

## Example Running Code

# Open AI Evaluate by Prompt
python -m opsec_l4_eval.eval_prompts_openai `
  --l4-root "C:\Users\sod\Desktop\DS680\DS680_Assignment_3\L4_SEC_AccountTakeoverDetection" `
  --prompt-judge "C:\Users\sod\Desktop\DS680\DS680_Assignment_3\L4_SEC_AccountTakeoverDetection\prompt_src\prompt_scoring_model.txt" `
  --model gpt-4o

python -m opsec_l4_eval.eval_prompts_openai `
  --l4-root "C:\Users\sod\Desktop\DS680\DS680_Assignment_3\L4_SEC_ContinuityRecovery" `
  --prompt-judge "C:\Users\sod\Desktop\DS680\DS680_Assignment_3\L4_SEC_ContinuityRecovery\prompt_src\prompt_scoring_model.txt" `
  --model gpt-4o

python -m opsec_l4_eval.eval_prompts_openai `
  --l4-root "C:\Users\sod\Desktop\DS680\DS680_Assignment_3\L4_SEC_EndpointScrapingEfficacy" `
  --prompt-judge "C:\Users\sod\Desktop\DS680\DS680_Assignment_3\L4_SEC_EndpointScrapingEfficacy\prompt_src\prompt_scoring_model.txt" `
  --model gpt-4o

python -m opsec_l4_eval.eval_prompts_openai `
  --l4-root "C:\Users\sod\Desktop\DS680\DS680_Assignment_3\L4_SEC_SecretsManagement" `
  --prompt-judge "C:\Users\sod\Desktop\DS680\DS680_Assignment_3\L4_SEC_SecretsManagement\prompt_src\prompt_scoring_model.txt" `
  --model gpt-4o

# Open AI Evaluate by Documment Analysis
python -m opsec_l4_eval.eval_docs_openai `
  --l4-root "../L4_SEC_AccountTakeoverDetection" `
  --doc-judge "../L4_SEC_AccountTakeoverDetection/doc_analysis/doc_scoring_model.txt" `
  --model gpt-4o

python -m opsec_l4_eval.eval_docs_openai `
  --l4-root "../L4_SEC_ContinuityRecovery" `
  --doc-judge "../L4_SEC_ContinuityRecovery/doc_analysis/doc_scoring_model.txt" `
  --model gpt-4o

python -m opsec_l4_eval.eval_docs_openai `
  --l4-root "../L4_SEC_EndpointScrapingEfficacy" `
  --doc-judge "../L4_SEC_EndpointScrapingEfficacy/doc_analysis/doc_scoring_model.txt" `
  --model gpt-4o

python -m opsec_l4_eval.eval_docs_openai `
  --l4-root "../L4_SEC_SecretsManagement" `
  --doc-judge "../L4_SEC_SecretsManagement/doc_analysis/doc_scoring_model.txt" `
  --model gpt-4o

# Deep Seek Evaluate by Prompt
python -m opsec_l4_eval.eval_prompts_deepseek `
  --l4-root "..\L4_SEC_AccountTakeoverDetection" `
  --prompt-judge "..\L4_SEC_AccountTakeoverDetection\prompt_src\prompt_scoring_model.txt" `
  --model-under-test "deepseek-chat" `
  --judge-model "deepseek-chat"

python -m opsec_l4_eval.eval_prompts_deepseek `
  --l4-root "..\L4_SEC_ContinuityRecovery" `
  --prompt-judge "..\L4_SEC_ContinuityRecovery\prompt_src\prompt_scoring_model.txt" `
  --model-under-test "deepseek-chat" `
  --judge-model "deepseek-chat"

python -m opsec_l4_eval.eval_prompts_deepseek `
  --l4-root "..\L4_SEC_EndpointScrapingEfficacy" `
  --prompt-judge "..\L4_SEC_EndpointScrapingEfficacy\prompt_src\prompt_scoring_model.txt" `
  --model-under-test "deepseek-chat" `
  --judge-model "deepseek-chat"

python -m opsec_l4_eval.eval_prompts_deepseek `
  --l4-root "..\L4_SEC_SecretsManagement" `
  --prompt-judge "..\L4_SEC_SecretsManagement\prompt_src\prompt_scoring_model.txt" `
  --model-under-test "deepseek-chat" `
  --judge-model "deepseek-chat"

# DeepSeek evaluate by Document Analysis
python -m opsec_l4_eval.eval_docs_deepseek `
  --l4-root "..\L4_SEC_AccountTakeoverDetection" `
  --doc-judge "..\L4_SEC_AccountTakeoverDetection\doc_analysis\doc_scoring_model.txt" `
  --model deepseek-chat

python -m opsec_l4_eval.eval_docs_deepseek `
  --l4-root "..\L4_SEC_ContinuityRecovery" `
  --doc-judge "..\L4_SEC_ContinuityRecovery\doc_analysis\doc_scoring_model.txt" `
  --model deepseek-chat

python -m opsec_l4_eval.eval_docs_deepseek `
  --l4-root "..\L4_SEC_EndpointScrapingEfficacy" `
  --doc-judge "..\L4_SEC_EndpointScrapingEfficacy\doc_analysis\doc_scoring_model.txt" `
  --model deepseek-chat

#Evaluate Results
  python .\src\opsec_l4_eval\analyze_l4_generic.py `
>>   "L4_SEC_AccountTakeoverDetection\prompt_scoring\openai_results\prompt_evaluation_results.json" `
>>   "L4_SEC_AccountTakeoverDetection\doc_analysis\doc_evaluation_results.json"

python .\src\opsec_l4_eval\analyze_l4_generic.py `
>>   "L4_SEC_EndpointScrapingEfficacy\prompt_scoring\openai_results\prompt_evaluation_results.json" `
>>   "L4_SEC_EndpointScrapingEfficacy\doc_analysis\doc_evaluation_results.json"

python .\src\opsec_l4_eval\analyze_l4_generic.py `
>>   "L4_SEC_EndpointScrapingEfficacy\prompt_scoring\openai_results\prompt_evaluation_results.json" `
>>   "L4_SEC_EndpointScrapingEfficacy\doc_analysis\doc_evaluation_results.json"

python .\src\opsec_l4_eval\analyze_l4_generic.py `
>>   "L4_SEC_SecretsManagement\prompt_scoring\openai_results\prompt_evaluation_results.json" `
>>   "L4_SEC_SecretsManagement\doc_analysis\doc_evaluation_results.json"
