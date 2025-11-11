from .dataclasses import Instance
from typing import Set, List, Dict
from collections import defaultdict


class Index:
    def __init__(self):
        self.contents_index: Dict[str, Set[str]] = defaultdict(set)    # content_key -> Set of Instance IDs containing this field
        self.metadata_index: Dict[str, Set[str]] = defaultdict(set)    # metadata_key -> Set of Instance IDs containing this metadata
        self.executors_index: Dict[str, Set[str]] = defaultdict(set)   # module_name -> Set of Instance IDs that have executed this module

    def query(
        self,
        content_keys: List[str] = None,
        metadata_keys: List[str] = None,
        executor_names: List[str] = None,
    ) -> Set[str]:
        content_keys = [] if content_keys is None else content_keys
        metadata_keys = [] if metadata_keys is None else metadata_keys
        executor_names = [] if executor_names is None else executor_names

        instance_ids = set()
        # Query by content keys
        for key in content_keys:
            if key in self.contents_index:
                if not instance_ids:
                    instance_ids = self.contents_index[key].copy()
                else:
                    instance_ids &= self.contents_index[key]
            else:
                return set()
        # Query by metadata keys
        for key in metadata_keys:
            if key in self.metadata_index:
                if not instance_ids:
                    instance_ids = self.metadata_index[key].copy()
                else:
                    instance_ids &= self.metadata_index[key]
            else:
                return set()
        # Query by executor names
        for name in executor_names:
            if name in self.executors_index:
                if not instance_ids:
                    instance_ids = self.executors_index[name].copy()
                else:
                    instance_ids &= self.executors_index[name]
            else:
                return set()
        return instance_ids

    def _add(self, fields: List[str], instance_id: str, index: Dict[str, Set[str]]) -> None:
        for field in fields:
            index[field].add(instance_id)

    def add(self, instance: Instance) -> None:
        for fields in [instance.contents, instance.metadata, instance.executors]:
            self._add(list(fields.keys()), instance.id, self.contents_index)

    def _remove(self, fields: List[str], instance_id: str, index: Dict[str, Set[str]]) -> None:
        for field in fields:
            if instance_id in index[field]:
                index[field].remove(instance_id)
                if not index[field]:
                    del index[field]

    def remove(self, instance: Instance) -> None:
        for fields in [instance.contents, instance.metadata, instance.executors]:
            self._remove(list(fields.keys()), instance.id, self.contents_index)

    def update(
        self, 
        instance_id: str, 
        new_contents: dict = None, 
        new_metadata: dict = None, 
        new_executors: dict = None,
    ) -> None:
        if new_contents is not None:
            self._add(list(new_contents.keys()), instance_id, self.contents_index)
        if new_metadata is not None:
            self._add(list(new_metadata.keys()), instance_id, self.metadata_index)
        if new_executors is not None:
            self._add(list(new_executors.keys()), instance_id, self.executors_index)

class DataPool:
    def __init__(self):
        self.index = Index()
        self.instances: Dict[str, Instance] = {}

    def __len__(self) -> int:
        return len(self.instances)

    def add(
        self, 
        data: List[dict] = None,
        instances: List[Instance] = None
    ) -> None:
        if data is not None:
            for contents in data:
                # Update self.instances
                instance = Instance(contents=contents)
                self.instances[instance.id] = instance
                # Update self.index
                self.index.add(instance)
        if instances is not None:
            for instance in instances:
                # Update self.instances
                self.instances[instance.id] = instance
                # Update self.index
                self.index.add(instance)

    def remove(self, instance_id: str) -> None:
        if instance_id in self.instances:
            # Update self.index
            instance = self.instances[instance_id]
            self.index.remove(instance)
            # Remove from self.instances
            del self.instances[instance_id]

    def update(self, instance_id: str, new_contents: dict, executed_module: str = None) -> None:
        assert instance_id in self.instances, f"Instance ID {instance_id} not found in DataPool."
        # Update self.instances
        self.instances[instance_id].contents.update(new_contents)
        if executed_module is not None:
            self.instances[instance_id].executors.update({executed_module: list(new_contents.keys())})
        # Update self.index
        self.index.update(
            instance_id, 
            new_contents=new_contents, 
            new_executors={executed_module: list(new_contents.keys())} if executed_module is not None else None
        )

    def retrieve(
        self, 
        target_contents: List[str] = None,
        target_modules: List[str] = None, 
        avoid_contents: List[str] = None, 
        avoid_modules: List[str] = None,
    ) -> List[Instance]:
        instance_ids = self.index.query(
            content_keys=target_contents,
            executor_names=target_modules
        ) if (target_contents or target_modules) else set(self.instances.keys())

        instance_ids -= self.index.query(content_keys=avoid_contents) if avoid_contents else set()
        instance_ids -= self.index.query(executor_names=avoid_modules) if avoid_modules else set()
        
        return [self.instances[instance_id] for instance_id in instance_ids]