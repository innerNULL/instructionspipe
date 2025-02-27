# -*- coding: utf-8 -*-
# file: cache.py
# date: 2025-02-27


import pdb
from typing import Dict, List, Optional


class CacheBase:
    def __init__(self):
        pass

    def write(self, key: str, val: str) -> None:
        return

    def read(self, key: str) -> Optional[None]:
        return

    def refresh_key(self, key: str) -> None:
        return 


class InMemCache(CacheBase):
    def __init__(self):
        self.data: Dict[str, str] = {}
        self.queue: List[str] = []
        self.size: int = -1

    @classmethod
    def new_with_configs(cls, configs: Dict):
        out = cls()
        out.size = configs["size"]
        return out

    def refresh_key(self, key: str) -> None:
        key_idx: Optional[None] = None
        if key in self.data:
            key_idx = self.queue.index(key) 
            self.queue = (
                [self.queue[key_idx]]
                + self.queue[0:key_idx] 
                + self.queue[key_idx + 1:]
            )
        else:
            self.queue = [key] + self.queue
            if len(self.queue) > self.size:
                rm_key: str = self.queue.pop()
                del self.data[rm_key]
                print("Key '%s' has been removed from cache" % rm_key)
        return 

    def write(self, key: str, val: str) -> None:
        self.refresh_key(key)
        self.data[key] = val
        return

    def read(self, key: str) -> Optional[str]:
        if key in self.data:
            self.refresh_key(key)
            print("Cache: Hit one")
        return self.data.get(key, None)
