import dataclasses
import hashlib
import logging
import os
import unittest
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from unittest import result

import hydra
from data.dali_core import InputPipelineCore
from data.dali_npy import NPyLegacyDataPipeline
from omegaconf import DictConfig, OmegaConf
from utils.app import MPIApplication
from utils.executor import get_executor_from_config


@dataclasses.dataclass
class PipelineResultEntry(object):
    d_hash: str
    l_hash: str

    in_local: Union[int, List[int]]
    in_instance: Optional[Union[int, List[int]]]
    in_global: int


def check_all_same(x: Sequence) -> bool:
    return all(x[0] == y for y in x)


class DataStagerTest(unittest.TestCase):
    # Application class that implements common logic
    class Application(MPIApplication):
        def run(self):
            pass

        def setup(self):
            super().setup()

            self.train_remote, self.val_remote = self._get_datasets(
                stage=False)
            self.train_local, self.val_local = self._get_datasets(stage=True)

            self.train_remote_iter = iter(self.train_remote)
            self.val_remote_iter = iter(self.val_remote)

            self.stager_executor = get_executor_from_config(self._config)
            with self.stager_executor:
                self.train_local.stage_data(executor=self.stager_executor)
                self.val_local.stage_data(executor=self.stager_executor)
            self._distenv.global_barrier()

            self.train_local_iter = iter(self.train_local)
            self.val_local_iter = iter(self.val_local)

        def validate_pipeline(self,
                              test_suite: unittest.TestCase,
                              pipeline: Iterable) -> Dict[str, PipelineResultEntry]:
            per_rank_result = self._gather_epoch_results(pipeline)
            global_results = self._exange_results(test_suite, per_rank_result)

            self._validate_result_correctness(test_suite, global_results)
            return global_results

        def _get_datasets(self, stage: bool = True) -> Tuple[InputPipelineCore, InputPipelineCore]:
            updated_config: DictConfig = self._config["data"].copy()

            if not stage:
                updated_config["stage"] = False
                updated_config["shard_type"] = "global"

            return NPyLegacyDataPipeline.build(updated_config,
                                               self._distenv,
                                               self._distenv.local_rank)

        def _gather_epoch_results(self, pipeline: Iterable) \
                -> Dict[str, PipelineResultEntry]:
            result_dict = {}
            for input_data in pipeline:
                data = input_data[0]["data"].cpu().numpy()
                label = input_data[0]["label"].cpu().numpy()

                d_hash = hashlib.sha512(data.tobytes()).hexdigest()
                l_hash = hashlib.sha512(label.tobytes()).hexdigest()

                if d_hash in result_dict:
                    assert l_hash == result_dict[d_hash].l_hash
                    result_dict[d_hash].in_local += 1
                else:
                    result_dict[d_hash] = PipelineResultEntry(d_hash, l_hash,
                                                              in_local=1,
                                                              in_instance=None,
                                                              in_global=None)

            self._distenv.global_barrier()
            return result_dict

        def _exange_results(self,
                            test_suite: unittest.TestCase,
                            rank_result: Dict[str, PipelineResultEntry]) \
                -> Dict[str, PipelineResultEntry]:

            result_dict_global = {}

            if not self._distenv.is_single:
                instances_map = self._distenv.master_mpi_comm.allgather(
                    self._distenv.instance)
                result_dict_hashes = self._distenv.master_mpi_comm.allgather(
                    list(rank_result.keys()))
            else:
                instances_map = [0]
                result_dict_hashes = [list(rank_result.keys())]
            mpi_rank = self._distenv.master_mpi_comm.Get_rank()
            for rank, hash_list in enumerate(result_dict_hashes):
                for hash in hash_list:
                    # if hash not in current rank result dict, add it with no occurence
                    l_hash = self._distenv.master_mpi_comm.bcast(
                        rank_result[hash].l_hash if hash in rank_result else None, rank)
                    if hash not in rank_result:
                        rank_result[hash] = PipelineResultEntry(
                            hash, l_hash, in_local=0, in_instance=0, in_global=0)

            # Exange data about occurences
            for hash_list in result_dict_hashes:
                for hash in hash_list:
                    all_gathered_in_local = self._distenv.master_mpi_comm.allgather(
                        rank_result[hash].in_local)

                    test_suite.assertTrue(check_all_same(
                        self._distenv.master_mpi_comm.allgather(
                            rank_result[hash].l_hash)),
                        f"Target hashes mismatched for input hash {hash[:16]}")

                    result_dict_global[hash] = PipelineResultEntry(rank_result[hash].d_hash,
                                                                   rank_result[hash].l_hash,
                                                                   in_local=all_gathered_in_local,
                                                                   in_instance=sum((x for x, i in zip(all_gathered_in_local,
                                                                                                      instances_map)
                                                                                    if i == self._distenv.instance)),
                                                                   in_global=sum(all_gathered_in_local))
            return result_dict_global

        def _validate_result_correctness(self,
                                         test_suite: unittest.TestCase,
                                         results: Dict[str, PipelineResultEntry]) -> None:
            for k, v in results.items():
                test_suite.assertEqual(
                    v.in_instance, 1, f"Item {k} was not readed {v.in_instance} times")
            test_suite.assertTrue(all((x.in_global == v.in_global for x in results.values())),
                                  "Not all elements was read globally the same amount of times")

        def _compare_two_results(self,
                                 test_suite: unittest.TestCase,
                                 result_a: Dict[str, PipelineResultEntry],
                                 result_b: Dict[str, PipelineResultEntry]) -> None:
            all_input_hashes = list(
                set(result_a.keys()) | set(result_b.keys()))

            for hash in all_input_hashes:
                test_suite.assertIn(
                    hash, result_a, f"Input hash {hash} not in result_a container")
                test_suite.assertIn(
                    hash, result_b, f"Input hash {hash} not in result_b container")

    # Test case setup classes

    @classmethod
    def setUpClass(cls) -> None:
        with hydra.initialize(config_path="../config"):
            cls._app_handler = cls.Application(
                config=hydra.compose(config_name=os.getenv("TEST_CONFIG", "baseline"),
                                     overrides=os.getenv("TEST_OVERRIDE", "").split(" ")))
            cls._app_handler.setup()

    # Actual test cases
    def test_every_train_remote_file_is_accessed_once_per_instance(self) -> None:
        self._app_handler._distenv.global_barrier()
        self._app_handler.validate_pipeline(
            self, self._app_handler.train_remote_iter)

    def test_every_val_remote_file_is_accessed_once_per_instance(self) -> None:
        self._app_handler._distenv.global_barrier()
        self._app_handler.validate_pipeline(
            self, self._app_handler.val_remote_iter)

    def test_every_train_local_file_is_accessed_once_per_instance(self) -> None:
        self._app_handler._distenv.global_barrier()
        self._app_handler.validate_pipeline(
            self, self._app_handler.train_local_iter)

    def test_every_val_local_file_is_accessed_once_per_instance(self) -> None:
        self._app_handler._distenv.global_barrier()
        self._app_handler.validate_pipeline(
            self, self._app_handler.val_local_iter)

    def test_train_remote_is_consistent_accross_epochs(self) -> None:
        self._app_handler._distenv.global_barrier()
        old_data = None
        for _ in range(3):
            current_data = self._app_handler.validate_pipeline(
                self, self._app_handler.train_remote_iter)
            if old_data is not None:
                self._app_handler._compare_two_results(
                    self, old_data, current_data)
            old_data = current_data

    def test_val_remote_is_consistent_accross_epochs(self) -> None:
        self._app_handler._distenv.global_barrier()
        old_data = None
        for _ in range(3):
            current_data = self._app_handler.validate_pipeline(
                self, self._app_handler.val_remote_iter)
            if old_data is not None:
                self._app_handler._compare_two_results(
                    self, old_data, current_data)
            old_data = current_data

    def test_train_local_is_consistent_accross_epochs(self) -> None:
        self._app_handler._distenv.global_barrier()
        old_data = None
        for _ in range(3):
            current_data = self._app_handler.validate_pipeline(
                self, self._app_handler.train_local_iter)
            if old_data is not None:
                self._app_handler._compare_two_results(
                    self, old_data, current_data)
            old_data = current_data

    def test_val_local_is_consistent_accross_epochs(self) -> None:
        self._app_handler._distenv.global_barrier()
        old_data = None
        for _ in range(3):
            current_data = self._app_handler.validate_pipeline(
                self, self._app_handler.val_local_iter)
            if old_data is not None:
                self._app_handler._compare_two_results(
                    self, old_data, current_data)
            old_data = current_data

    def test_train_local_are_same_as_remote(self) -> None:
        self._app_handler._distenv.global_barrier()
        local_data = self._app_handler.validate_pipeline(
            self, self._app_handler.train_local_iter)
        remote_data = self._app_handler.validate_pipeline(
            self, self._app_handler.train_remote_iter)

        self._app_handler._compare_two_results(self, local_data, remote_data)

    def test_val_local_are_same_as_remote(self) -> None:
        self._app_handler._distenv.global_barrier()
        local_data = self._app_handler.validate_pipeline(
            self, self._app_handler.val_local_iter)
        remote_data = self._app_handler.validate_pipeline(
            self, self._app_handler.val_remote_iter)

        self._app_handler._compare_two_results(self, local_data, remote_data)


if __name__ == "__main__":
    unittest.main()
