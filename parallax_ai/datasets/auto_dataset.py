from .base_class import Dataset
from .safeguarding import SEALSBenchDataset, PKUSafeRLHFQADataset, SEASafeguardBenchDataset


AVAILABLE_DATASETS = [
    "seals_bench",
    "pku_saferlhf_qa",
    "sea_safeguard_bench",
]

class AutoDataset:
    available_datasets = {
        "seals_bench": SEALSBenchDataset,
        "pku_saferlhf_qa": PKUSafeRLHFQADataset,
        "sea_safeguard_bench": SEASafeguardBenchDataset,
    }

    @classmethod
    def from_name(cls, dataset_name: str) -> Dataset:
        if dataset_name not in cls.available_datasets:
            raise ValueError(f"No dataset found for dataset name: {dataset_name}")
        return cls.available_datasets[dataset_name]