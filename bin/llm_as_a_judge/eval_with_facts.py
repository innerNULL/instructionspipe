# -*- coding: utf-8 -*-
# file: eval_with_facts.py
# date: 2025-01-21


import os
import pdb
import traceback
import sys
import json
import asyncio
import duckdb
from tqdm import tqdm
from typing import Union, Optional, List, Dict, Coroutine, Callable, Any
from openai import AsyncOpenAI, AsyncAzureOpenAI
from openai import ChatCompletion
from pydantic import BaseModel


SQL: str = \
"""
with
eval_results as (
  select *
  from 
  read_json(
    '__INF_RESULTS_PATH__', 
    auto_detect=true, format='newline_delimited'
  )
),
cleaned_eval_results as (
  select
  factuality, 
  eligibility
  from 
  eval_results
),
eval_avg_metrics as (
  select 
  sum(factuality) / count(1) as avg_factuality,
  sum(eligibility) / count(1) as avg_eligibility
  from 
  cleaned_eval_results
)
select * from eval_avg_metrics;
""".strip("\n")


SYS_PROMPT_FACTUALITY_SCORE_RESP_LEVEL: str = \
"""
## Your Role
You're a document verification expert. 

## Your Task 
Check whether all the information in Response can be:
* Fully Supported and contained by Evidence.
* Totally accurate according to the Evidence (and Instruction)

## Evidence
__CONTEXT__

## Instruction
__USER_QUERY__

## Response
__RESPONSE__

## Expected Output
You must output only a JSON which contains following keys in order:
* rationale: A brief explanation for the assigned label.
* label: One of 'supported', 'unsupported', 'contradictory'
""".strip("\n")


SYS_PROMPT_FACTUALITY_SCORE_RESP_LEVEL_FACTS: str = \
"""
Your task is to check if the Response is accurate to the Evidence.
Generate ’Accurate’ if the Response is accurate when verified according to the Evidence,
or ’Inaccurate’ if the Response is inaccurate (contradicts the evidence) or cannot be
verified.
**Query**:\n\n__USER_QUERY__\n\n**End of Query**\n
**Evidence**\n\n__CONTEXT__\n\n**End of Evidence**\n
**Response**:\n\n__RESPONSE__\n\n**End of Response**\n

You must output only a JSON which contains following keys:
* rationale: A brief explanation for the assigned label.
* label: One of 'supported', 'unsupported', 'contradictory'
""".strip("\n")


SYS_PROMPT_FACTUALITY_SCORE_SENT_LEVEL_FACTS: str = \
"""
You are a helpful and harmless AI assistant. You will be provided with a textual context
and a model-generated response.

Your task is to analyze the response sentence by sentence and classify each sentence
according to its relationship with the provided context.

**Instructions:**
1. **Decompose the response into individual sentences.**
2. **For each sentence, assign one of the following labels:**
* **‘supported‘**: The sentence is entailed by the given context. Provide a
supporting excerpt from the context. The supporting except must *fully* entail the
sentence. If you need to cite multiple supporting excepts, simply concatenate them.
* **‘unsupported‘**: The sentence is not entailed by the given context. No excerpt is
needed for this label.
* **‘contradictory‘**: The sentence is falsified by the given context. Provide a
contradicting excerpt from the context.
* **‘no_rad‘**: The sentence does not require factual attribution (e.g., opinions,
greetings, questions, disclaimers). No excerpt is needed for this label.
3. **For each label, provide a short rationale explaining your decision.** The rationale
should be separate from the excerpt.
4. **Be very strict with your ‘supported‘ and ‘contradictory‘ decisions.** Unless you
can find straightforward, indisputable evidence excerpts *in the context* that a
sentence is ‘supported‘ or ‘contradictory‘, consider it ‘unsupported‘. You should not
employ world knowledge unless it is truly trivial.

**Input Format:**
The input will consist of two parts, clearly separated:
* **Context:** The textual context used to generate the response.
* **Response:** The model-generated response to be analyzed.

**Output Format:**
For each sentence in the response, output a JSON object with the following fields:
* ‘"sentence"‘: The sentence being analyzed.
* ‘"label"‘: One of ‘supported‘, ‘unsupported‘, ‘contradictory‘, or ‘no_rad‘.
* ‘"rationale"‘: A brief explanation for the assigned label.
* ‘"excerpt"‘: A relevant excerpt from the context. Only required for ‘supported‘ and ‘
contradictory‘ labels.
Output each JSON object on a new line.
**Example:**
**Input:**
'''
Context: Apples are red fruits. Bananas are yellow fruits.
Response: Apples are red. Bananas are green. Bananas are cheaper than apples. Enjoy your
fruit!
'''
**Output:**
{"sentence": "Apples are red.", "label": "supported", "rationale": "The context
explicitly states that apples are red.", "excerpt": "Apples are red fruits."}
{"sentence": "Bananas are green.", "label": "contradictory", "rationale": "The context
states that bananas are yellow, not green.", "excerpt": "Bananas are yellow fruits."}
{"sentence": "Bananas are cheaper than apples.", "label": "unsupported", "rationale": "
The context does not mention the price of bananas or apples.", "excerpt": null}
{"sentence": "Enjoy your fruit!", "label": "no_rad", "rationale": "This is a general
expression and does not require factual attribution.", "excerpt": null}
**Now, please analyze the following context and response:**
**User Query:**
__USER_QUERY__
**Context:**
__CONTEXT__
**Response:**
__RESPONSE__
""".strip("\n")


SYS_PROMPT_FACTUALITY_SCORE_RESP_LEVEL_SELF_REFINE: str = \
"""
Your task is to verify whether a given sentence is entailed by a given context or not.
Answer only in YES or NO without any additional text. Do not try to avoid answering, or
apologize, or give any answer that isn’t simply YES or NO.
**Sentence**
{json_dict["sentence"]}
**Context**
{json_dict["excerpt"]}
""".strip("\n")


SYS_PROMPT_ELIGIBILITY_SCORE: str = \
"""
# Rubrics
Your mission is to judge the response from an AI model (the *test* response) 
according to the given input (context).

Please use the following rubric criteria to judge the responses:
<START OF RUBRICS>
Your task is to analyze the test response based on the criterion of "Instruction
Following". Start your analysis with "Analysis".

**Instruction Following**
Please first list the instructions in the user query.
In general, an instruction is VERY important if it is specifically asked for in the
prompt and deviates from the norm. Please highlight such specific keywords.
You should also derive the task type from the user query and include the task-specific
implied instructions.
Sometimes, no instruction is available in the user query.
It is your job to infer if the instruction is to autocomplete the user query or is
asking the LLM for follow-ups.
After listing the instructions, you should rank them in order of importance.
of the instructions.
You should itemize, for each instruction, whether the response meets, partially meets,
or does not meet the requirement, using reasoning.
You should start reasoning first before reaching a conclusion about whether the response
satisfies the requirement.
Citing examples while reasoning is preferred.
Reflect on your answer and consider the possibility that you are wrong.
If you are wrong, explain clearly what needs to be clarified, improved, or changed in
the rubric criteria and guidelines.
In the end, express your final verdict as one of the following three json objects:

```json
{
"Instruction Following": "No Issues"
}
```
```json
{
"Instruction Following": "Minor Issue(s)"
}
```
```json
{
"Instruction Following": "Major Issue(s)"
}
```
<END OF RUBRICS>

# Your task
## User query
<|begin_of_query|>
__USER_QUERY__
<|end_of_query|>

## Test Response:
<|begin_of_test_response|>
__RESPONSE__
<|end_of_test_response|>

## Given Context:
<|begin_of_given_context|>
__CONTEXT__
<|end_of_given_context|>

Please write your analysis and final verdict for the test response.
""".strip("\n")


class FactsInput(BaseModel):
    src_text: str
    gen_text: str
    instruction: Optional[str] = None
    gt_factuality: Optional[float] = None
    gt_eligibility: Optional[float] = None


class Judgement(BaseModel):
    name: Optional[str] = None
    score: Optional[float] = None
    result: Optional[str] = None
    rationale: Optional[str] = None
    inputs: Optional[FactsInput] = None
    msgs: Optional[str] = None


class Judgements(BaseModel):
    name: Optional[str] = None
    score: Optional[float] = None
    outputs: Optional[List[Judgement]] = None


class MultiJudgements(BaseModel):
    factuality: Optional[float] = None
    eligibility: Optional[float] = None
    factuality_rationales: Optional[List[str]] = None
    eligibility_rationales: Optional[List[str]] = None
    details: Optional[List[Judgements]] = None


class LlmCli:
    def __init__(self):
        self.cli: Optional[OpenAI | AsyncAzureOpenAI] = None
        self.async_cli: Optional[AsyncOpenAI] = None
        self.model: Optional[str] = None
        self.seed: Optional[int] = None
        self.temperature: Optional[float] = None
        self.top_p: Optional[float] = None

    @classmethod
    def new(
        cls, 
        model: str, 
        api_key: str,
        api_url: str, 
        seed: int=2, 
        api_type: str="openai",
        api_version: str="",
        temperature: float=0.0, 
        top_p: float=0.01
    ):
        out = cls()
        if api_type == "openai":
            out.async_cli = AsyncOpenAI(api_key=api_key, base_url=api_url)
        # Refer to https://elanthirayan.medium.com/using-azure-openai-with-python-a-step-by-step-guide-415d5850169b
        elif api_type == "azure":
            out.async_cli = AsyncAzureOpenAI(
                azure_endpoint=api_url, 
                api_version=api_version,
                api_key=api_key
            )
        out.model = model
        out.seed = seed
        out.temperature = temperature
        out.top_p = top_p
        return out

    @classmethod
    def new_with_configs(cls, configs: Dict):
        return cls.new(
            model=configs["model"], 
            api_key=configs["api_key"],
            api_url=configs["api_url"],
            temperature=configs["temperature"],
            seed=configs["seed"],
            api_type=configs["api_type"],
            api_version=configs["api_version"]
        )

    async def async_run(
        self, 
        msgs: Union[Dict, List[Dict]],
        json_schema: Optional[Dict]=None,
        temperature: Optional[float]=None, 
        max_tokens: int=4096
    ) -> Optional[ChatCompletion]:
        if isinstance(msgs, dict):
            msgs = [msgs]
        for msg in msgs:
            if msg["content"] is None:
                return None

        return await self.async_cli.chat.completions.create(
            model=self.model, 
            messages=msgs,
            seed=self.seed,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=max_tokens,
            top_p=self.top_p,
            response_format=json_schema
        )


class FactsMetrics:
    def __init__(self):
        self.llms: Optional[Dict[str, LlmCli]] = None
        self.factuality_sys_prompt_temp: Optional[str] = None
        self.factuality_self_refine_sys_prompt_temp: Optional[str] = None
        self.eligibality_sys_prompt_temp: Optional[str] = None

    @classmethod
    def new_with_llms(cls, 
        llms: Dict[str, LlmCli], 
        factuality_sys_prompt_temp: str,
        factuality_self_refine_sys_prompt_temp: str,
        eligibility_sys_prompt_temp: str
    ):
        out = cls()
        out.llms = llms
        out.factuality_sys_prompt_temp = factuality_sys_prompt_temp
        out.factuality_self_refine_sys_prompt_temp = \
            factuality_self_refine_sys_prompt_temp
        out.eligibility_sys_prompt_temp = eligibility_sys_prompt_temp
        return out

    async def run(self, facts_input: FactsInput) -> Judgement:
        out: MultiJudgements = \
            await self.run_multi_judgements(facts_input)
        #pdb.set_trace()
        return out

    async def run_factuality_score(
        self, 
        facts_input: FactsInput, 
        model: str
    ) -> Judgement:
        msgs: List[Dict] = []
        user_msg_1 = {
            "role": "user",
            "content": self.build_factuality_sys_prompt(facts_input)
        }
        msgs.append(user_msg_1)
        print("Run init factuality eval")
        resp_1 = await self.llms[model].async_run(msgs)
        result_1: str = llm_json_clean(resp_1.choices[0].message.content)
        msgs.append({"role": "assistant", "content": result_1})
        """
        user_msg_2 = {
            "role": "user", 
            "content": self.build_factuality_self_refine_prompt(facts_input) 
        }
        msgs.append(user_msg_2)
        print("Run factuality eval double check")
        resp_2 = await self.llms[model].async_run(msgs)
        result_2: str = resp_2.choices[0].message.content
        msgs.append({"role": "assistant", "content": result_2})
        """

        out: Judgement = Judgement()
        out.msgs = json.dumps(msgs)
        out.name = "factuality"
        try: 
            score_1: float = (
                1.0 if json.loads(result_1)["label"] == "supported" 
                else 0.0
            )
            """
            score_2: float = 1.0 if "yes" in result_2.lower() else 0.0
            score: float = (score_1 + score_2) / 2
            """
            score: float = score_1
            out.score = score
            out.result = json.loads(result_1)["label"]
            out.rationale = result_1
        except Exception as e:
            print("Sth wrong")
            print(traceback.format_exc())
            out.score = 0.5
        return out

    async def run_eligibility_score(
        self, 
        facts_input: FactsInput, 
        model: str
    ) -> Judgement:
        sys_prompt = self.eligibility_sys_prompt_temp
        sys_prompt = sys_prompt.replace("__USER_QUERY__", facts_input.instruction)
        sys_prompt = sys_prompt.replace("__CONTEXT__", facts_input.src_text)
        sys_prompt = sys_prompt.replace("__RESPONSE__", facts_input.gen_text)
        msgs: List[Dict] = [{"role": "user", "content": sys_prompt}]
        resp: str = (await self.llms[model].async_run(msgs)).choices[0].message.content
        msgs.append({"role": "assistant", "content": resp})
        print("Run eligibility eval")
        rationale: str = resp
        out: Judgement = Judgement()
        out.name = "eligibility"
        try:
            result: str = json.loads(
                resp.split("```json")[-1].replace("```", "")
            )["Instruction Following"]
            score: float = 0.0
            if result == "Minor Issue(s)":
                score = 0.25
            if result == "No Issues":
                score = 1.0
            else:
                pass
            out.result = result
            out.score = score
            out.rationale = rationale
            out.msgs = json.dumps(msgs)
        except Exception as e:
            print("Sth wrong")
            print(resp)
            print(traceback.format_exc())
            out.score = 0.5
        return out

    async def run_multi_judgements(
        self, 
        facts_input: FactsInput
    ) -> List[Judgements]:
        multi_judgements: List[Judgements] = []
        funcs: List[Callable] = [
            self.run_factuality_score,
            self.run_eligibility_score
        ]
        for func in funcs:
            tasks: List[Coroutine] = []
            for model in self.llms:
                tasks.append(func(facts_input, model))
            results: List[Optional[Judgement]] = await asyncio.gather(*tasks)
            judgements: Judgements = judgements_init(results)
            multi_judgements.append(judgements)
        out: MultiJudgements = MultiJudgements()
        for judgements in multi_judgements:
            if judgements.name == "factuality":
                out.factuality = judgements.score
                out.factuality_rationales = [
                    x.rationale for x in judgements.outputs
                ]
            if judgements.name == "eligibility":
                out.eligibility = judgements.score
                out.eligibility_rationales = [
                    x.rationale for x in judgements.outputs
                ]
        out.details = multi_judgements
        return out

    def build_factuality_sys_prompt(self, facts_input: FactsInput) -> str:
        out: str = self.factuality_sys_prompt_temp
        if facts_input.instruction is not None:
            out = out.replace("__USER_QUERY__", facts_input.instruction)
        else:
            out = out.replace("__USER_QUERY__", "N/A")
        out = out.replace("__CONTEXT__", facts_input.src_text)
        out = out.replace("__RESPONSE__", facts_input.gen_text)
        return out
    
    def build_factuality_self_refine_prompt(self, facts_input: FactsInput) -> str:
        out: str = self.factuality_self_refine_sys_prompt_temp
        return out


def judgements_init(results: List[Optional[Judgement]]) -> Judgements:
    results = [x for x in results if x is not None]
    out: Judgements = Judgements()
    out.outputs = results
    out.score = sum([x.score for x in results]) / len(results)
    out.name = results[0].name
    return out


def llm_json_clean(llm_json: str) -> str:
    return llm_json\
        .replace("```json", "")\
        .replace("```", "")


async def llms_init(configs: List[Dict]) -> Dict[str, LlmCli]:
    out: Dict[str, LlmCli] = {}
    for config in configs:
        model: str = config["model"]
        out[model] = LlmCli.new_with_configs(config)
    for model, cli in out.items():
        await cli.async_run({"role": "user", "content": "hi"})
    return out


def facts_input_load(
    input_data: List[Dict], 
    src_text_col: str, 
    out_text_col: str, 
    instruction_col: str, 
    gt_factuality_col: Optional[str] = None, 
    gt_eligibility_col: Optional[str] = None
) -> List[FactsInput]:
    out: List[FactsInput] = []
    for data in input_data:
        src_text: str = data[src_text_col]
        out_text: str = data[out_text_col]
        instruction: str = data.get(instruction_col, None)
        gt_factuality: Optional[float] = data.get(gt_factuality_col, None)
        gt_eligibility: Optional[float] = data.get(gt_eligibility_col, None)
        
        if src_text is None:
            src_text = ""
        if out_text is None:
            out_text = ""
        facts_input: FactsInput = FactsInput(
            src_text=src_text,
            gen_text=out_text,
            instruction=instruction,
            gt_factuality=gt_factuality, 
            gt_eligibility=gt_eligibility
        )
        out.append(facts_input)
    return out


async def main() -> None:
    configs: Dict = json.loads(open(sys.argv[1], "r").read())
    in_data_path: str = configs["in_data_path"]
    out_data_path: str = configs["out_data_path"]
    inf_results: List[Dict] = []
    if not os.path.exists(out_data_path):
        llms: Dict[str] = await llms_init(configs["llms"])
        facts_metrics: FactsMetrics = FactsMetrics.new_with_llms(
            llms, 
            SYS_PROMPT_FACTUALITY_SCORE_RESP_LEVEL,
            SYS_PROMPT_FACTUALITY_SCORE_RESP_LEVEL_SELF_REFINE,
            SYS_PROMPT_ELIGIBILITY_SCORE
        )
        raw_samples: List[Dict] = [
            json.loads(x) for x in open(in_data_path, "r").read().split("\n")
            if x not in {""}
        ][:configs["max_sample_size"]]
        for raw_sample in tqdm(raw_samples):
            sample: FactsInput = facts_input_load(
                [raw_sample], 
                configs["in_text_field"], 
                configs["out_text_field"], 
                configs["instruction_field"],
                configs["gt_factuality_field"],
                configs["gt_eligibility_field"]
            )[0]
            multi_judgements: MultiJudgements = await facts_metrics.run(sample)
            factuality: float = multi_judgements.factuality
            eligibility: float = multi_judgements.eligibility
            factuality_rationales: List[str] = multi_judgements.factuality_rationales
            eligibility_rationales: List[str] = multi_judgements.eligibility_rationales
            gt_factuality: Optional[float] = sample.gt_factuality
            gt_eligibility: Optional[float] = sample.gt_eligibility
            if gt_factuality is not None or gt_eligibility is not None:
                if gt_factuality is not None:
                    assert(factuality <= gt_factuality + 0.15)
                if gt_eligibility is not None:
                    assert(eligibility <= gt_eligibility + 0.15)
                print("Passed one test case")
            out: Dict = {
                "factuality": factuality, 
                "eligibility": eligibility,
                "gt_factuality": gt_factuality, 
                "gt_eligibility": gt_eligibility,
                "src_text": sample.src_text, 
                "gen_text": sample.gen_text,
                "instruction": sample.instruction,
                "factuality_rationales": factuality_rationales, 
                "eligibility_rationales": eligibility_rationales
            }
            inf_results.append(out)
        out_file = open(out_data_path, "w")
        for result in inf_results:
            out_file.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )
        out_file.close()
        print("Dumped the evaluation results to %s" % configs["out_data_path"])
    else:
        print("Inference result '%s' exists" % out_data_path)
        inf_results = [
            json.loads(x) for x in open(out_data_path, "r").read().split("\n")
            if x not in {""}
        ]
    sql: str = SQL.replace("__INF_RESULTS_PATH__", out_data_path)
    print(duckdb.query(sql))
    return


if __name__ == "__main__":
    asyncio.run(main())
