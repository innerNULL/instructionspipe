# -*- coding: utf-8 -*-
# file: run_md_example_gen.py
# date: 2025-04-10


import os
import sys
import pdb
import json
import asyncio
import io
import fitz
import base64
from pydantic import BaseModel
from typing import Dict, List, Optional, Annotated, Literal, Callable, Union, Type
from langchain_core.messages import AIMessage
from langchain_core.messages import AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel.io import AddableValuesDict
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pdf2image import convert_from_path
from tqdm import tqdm
from PIL import Image


TEMP_GEN_USER_MSG: str = \
"""
You need finish your task based on following descriptions.

## Role
You're a report planner. 
With a given example report, you need abstract it into sections and key elements.

## Task
You need generate an instruction about how to generate a similar report 
for given EHR data. 
You need tell which sections should be contained, and in each section, 
which key elements should be contained

## Output Format
Your should generate list of sections.
Each section should contains the key elements.
Each key elements has following attribute to declare:
* Descripion: Which kind of information this element should contain
* Format: Format of the element when it's under corresponding section.

The output shoud be enclosed by <answer> and </answer>.
""".strip("\n")


class State(BaseModel):
    llms: Dict[str, BaseChatModel] = {}
    agent_pdf2md_msgs: List[AnyMessage] = []
    agent_plan_gen_msgs: List[AnyMessage] = [] 


class Config(BaseModel):
    pdf_path: str
    llm_pdf2md: str = "gpt-4o"
    llm_plan_gen: str = "gpt-4o-mini"


def langchain_init(configs: Dict) -> None:
    os.environ["LANGSMITH_TRACING"] = str(configs["langsmith_tracing"]).lower()
    os.environ["LANGSMITH_ENDPOINT"] = configs["langsmith_endpoint"]
    os.environ["LANGSMITH_API_KEY"] = configs["langsmith_api_key"]
    os.environ["LANGSMITH_PROJECT"] = configs["langsmith_project"]


def pdf2image(
    path: str, merge: bool
) -> Union[List[Image.Image], Image.Image]:
    """
    Args:
    path: PDF file path
    merge: 
        If equals `True`, return a single image which contains all pages of PDF, 
        else return a list of page's images
    
    Returns:
    A list of images if merge is False, 
    otherwise a single merged image containing all pages.
    """
    
    # Open the PDF
    doc = fitz.open(path)
    
    # List to store page images
    images: List[Image.Image] = []
    
    # Loop through all pages and convert them to images
    for i in range(doc.page_count):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=fitz.Matrix(600 / 72, 600 / 72))
        #pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    
    # If merge is True, concatenate images
    if merge:
        total_height = sum(img.height for img in images)
        max_width = max(img.width for img in images)
        
        # Create a new image to fit all pages
        concatenated_image = Image.new("RGB", (max_width, total_height))
        
        # Paste each image into the new image
        current_y = 0
        for img in images:
            concatenated_image.paste(img, (0, current_y))
            current_y += img.height
        
        return concatenated_image
    else:
        return images


def image_to_base64(image: Image.Image) -> str:
    # Create a BytesIO object to hold the image bytes
    buffered: io.BytesIO = io.BytesIO()
    
    # Save the image to the BytesIO object in the desired format (e.g., PNG)
    image.save(buffered, format="PNG")
    
    # Get the byte data from the buffer
    img_bytes: bytes = buffered.getvalue()
    
    # Encode the byte data to base64
    img_base64: str = base64.b64encode(img_bytes).decode("utf-8")
    
    return img_base64


def images_to_base64(images: List[Image.Image]) -> List[str]:
    out: List[str] = []
    for i, image in enumerate(images):
        out.append(image_to_base64(image))
    return out


def pdf2base64(path: str) -> List[str]:
    return images_to_base64(pdf2image(path, False)) 


def pdf2md(llm: BaseChatModel, path: str) -> str:
    pdf_imgs: List[str] = pdf2base64(path) 
    msg_content: List[Dict] = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{x}"},
        } 
        for x in pdf_imgs 
    ] 
    msg_content = [
        {
            "type": "text", 
            "text": "Convert given images into a markdown report."
        }
    ] + msg_content 
    human_msg: HumanMessage = HumanMessage(content=msg_content)
    resp: AIMessage = llm.invoke([human_msg])
    return resp.content 


def agent_pdf2md(state: State, config: Config) -> Dict:
    cus_conf: Dict = config["configurable"]
    model: str = cus_conf["llm_pdf2md"]
    pdf_path: str = cus_conf["pdf_path"]
    pdf_imgs: List[str] = pdf2base64(pdf_path)
    llm: BaseChatModel = state.llms[model] 
    msg_content: List[Dict] = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{x}"},
        } 
        for x in pdf_imgs 
    ] 
    msg_content = [
        {
            "type": "text", 
            "text": "Convert given images into a markdown report."
        }
    ] + msg_content 
    human_msg: HumanMessage = HumanMessage(content=msg_content)
    msgs: List[AnyMessage] = [human_msg]
    resp: AIMessage = llm.invoke(msgs)
    msgs.append(resp)
    return {"agent_pdf2md_msgs": msgs}


def agent_plan_gen(state: State, config: Config) -> Dict:
    cus_conf: Dict = config["configurable"]
    model: str = cus_conf["llm_plan_gen"]
    llm: BaseChatModel = state.llms[model] 
    msgs: List[Dict] = []
    sys_msg: SystemMessage = SystemMessage(
        content=TEMP_GEN_USER_MSG
    )
    human_msg: HumanMessage = HumanMessage(
        content=state.agent_pdf2md_msgs[-1].content
    )
    msgs.append(sys_msg)
    msgs.append(human_msg)
    resp: AIMessage = llm.invoke(msgs) 
    return {"agent_plan_gen_msgs": msgs + [resp]}


def graph_build(
    state: Type,
    config: Type,
    agent_pdf2md: Callable,
    agent_plan_gen: Callable
) -> CompiledStateGraph:
    builder: StateGraph = StateGraph(state, config_schema=config)
    builder.add_node("agent_pdf2md", agent_pdf2md)
    builder.add_node("agent_plan_gen", agent_plan_gen)
    builder.add_edge(START, "agent_pdf2md")
    builder.add_edge("agent_pdf2md", "agent_plan_gen") 
    builder.add_edge("agent_plan_gen", END)
    graph: CompiledStateGraph = builder.compile()
    return graph


async def main() -> None:
    configs: Dict = json.loads(open(sys.argv[1], "r").read())
    print(configs)
    llm_configs: Dict = configs["llms"] 
    in_data_path: str = configs["in_data_path"]

    langchain_init(configs["langchain"])
    llms: Dict[str, BaseChatModel] = {
        x["model"]: init_chat_model(**x) for x in llm_configs
    } 
    graph: CompiledStateGraph = graph_build(
        State, 
        Config, 
        agent_pdf2md,
        agent_plan_gen
    )
    graph_configs: Config = {
        "configurable": Config(pdf_path=in_data_path).dict()
    }
    inputs: Dict = {"llms": llms}
    result: AddableValuesDict = graph.invoke(inputs, config=graph_configs)
    pdb.set_trace()
    return


if __name__ == "__main__":
    asyncio.run(main())
