# -*- coding: utf-8 -*-
# file: llm_cli.py
# date: 2024-12-09


import pdb
import hashlib
import json
from typing import Union, Optional, List, Dict, Coroutine, Callable, Any
from openai import AsyncOpenAI, AsyncAzureOpenAI
from openai.types.chat.chat_completion import ChatCompletion

from .cache import CacheBase
from .cache import InMemCache
from .logger import get_logger


LOGGER = get_logger(__name__)


class LlmCli:
    def __init__(self):
        self.cli: Optional[OpenAI | AsyncAzureOpenAI] = None
        self.async_cli: Optional[AsyncOpenAI] = None
        self.model: Optional[str] = None
        self.seed: Optional[int] = None
        self.temperature: Optional[float] = None
        self.top_p: Optional[float] = None
        self.cache: Optional[CacheBase] = None

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
        top_p: float=0.01,
        cache: Optional[CacheBase]=None
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
        if cache is None:
            out.cache = InMemCache.new_with_configs({"size": 10000})
        else:
            out.cache = cache
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
        msg: Union[str, Dict],
        prefix: Union[Dict, List[Dict]]=None,
        json_schema: Optional[Dict]=None,
        temperature: Optional[float]=None, 
        max_tokens: int=4096
    ) -> Optional[ChatCompletion]:
        if isinstance(msg, str):
            msg = {"role": "user", "content": msg}
        if prefix is None:
            prefix = []
        if isinstance(prefix, dict):
            prefix = [prefix]
        
        # TODO: 
        # TBH this logic is not very solid, is better to have a 
        # independent function to judge if skip of not.
        # Even skip can have 2 conditions: 
        #   1. Return 'N/A'
        #   2. Return row input based on some rules
        if msg["content"] is None:
            LOGGER.info("Return `None` as `msg[\"content\"] is None`")
            return None
        
        chaml: List[Dict] = prefix + [msg]
        cache_key: str = (
            hashlib.sha256(json.dumps(chaml).encode('utf-8')).hexdigest()
        )
        cache_val: Optional[str] = self.cache.read(cache_key)
        out: Optional[ChatCompletion] = None

        if cache_val is not None:
            LOGGER.info("Using cache for prefix \"{}\"".format(chaml))
            out = ChatCompletion.parse_raw(cache_val)
        else:
            out = await self.async_cli.chat.completions.create(
                model=self.model, 
                messages=prefix + [msg],
                seed=self.seed,
                temperature=self.temperature if temperature is None else temperature,
                max_tokens=max_tokens,
                top_p=self.top_p,
                # TODO:
                # Remove this as reasoning models (e.g. DS-R1) does not support constraint decoding
                #response_format=json_schema
            )
            LOGGER.info("Gen text by **{}** for msgs \"{}\"".format(self.model, chaml))
            self.cache.write(cache_key, out.to_json())
        return out
