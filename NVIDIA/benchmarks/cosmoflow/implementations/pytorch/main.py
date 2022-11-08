# Copyright (c) 2021-2022 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
from typing import Any
import utils

from data.dali_npy import NPyLegacyDataPipeline
from data.dali_tfr_gzip import TFRecordDataPipeline
from data.dali_synthetic import SyntheticDataPipeline
from utils.app import PytorchApplication
from omegaconf import OmegaConf

import hydra
import torch
import torch.distributed

from torch.nn.parallel import DistributedDataParallel as DDP
from utils.executor import get_executor_from_config

from model.cosmoflow import get_standard_cosmoflow_model, Convolution3DLayout

from trainer import Trainer
from optimizer import get_optimizer


class CosmoflowMain(PytorchApplication):
    def setup(self) -> None:
        super().setup()

        with utils.ProfilerSection("initialization", profile=self._config["profile"]):
            utils.logger.event(key=utils.logger.constants.CACHE_CLEAR)
            utils.logger.start(key=utils.logger.constants.INIT_START)

            number_of_nodes = self._distenv.size / self._distenv.local_size

            utils.logger.event(key=utils.logger.constants.SUBMISSION_BENCHMARK,
                               value="cosmoflow")
            utils.logger.event(key=utils.logger.constants.SUBMISSION_ORG,
                               value="NVIDIA")
            utils.logger.event(key=utils.logger.constants.SUBMISSION_DIVISION,
                               value="closed")
            utils.logger.event(key=utils.logger.constants.SUBMISSION_STATUS,
                               value="onprem")
            utils.logger.event(key=utils.logger.constants.SUBMISSION_PLATFORM,
                               value=f"{number_of_nodes}xNVIDIA DGX A100")

            utils.logger.event(key="number_of_nodes",
                               value=self._distenv.size // self._distenv.local_size)
            utils.logger.event(key="accelerators_per_node",
                               value=self._distenv.local_size)

            model_cfg = self._config["model"]
            train_cfg = model_cfg["training"]

            if "seed" in self._config:
                seed = self._config["seed"] + self._distenv.instance
            else:
                seed = random.randint(0, 65536)
                if not self._distenv.is_single:
                    seed = self._distenv.master_mpi_comm.bcast(
                        seed, root=0) + self._distenv.instance

            assert (train_cfg["weight_decay"] == 0.0 or train_cfg["dropout_rate"] ==
                    0.0), "Both 'weight_decay' and 'dropout_rate' cannot be different from 0"

            if self._config["data"]["dataset"] == "synthetic":
                self._training_pipeline = SyntheticDataPipeline(config=self._config["data"],
                                                                distenv=self._distenv,
                                                                sample_count=self._config["data"]["train_samples"],
                                                                device=self._distenv.local_rank)
                self._validation_pipeline = SyntheticDataPipeline(config=self._config["data"],
                                                                  distenv=self._distenv,
                                                                  sample_count=self._config["data"]["valid_samples"],
                                                                  device=self._distenv.local_rank)
            elif self._config["data"]["dataset"] == "cosmoflow_npy":
                self._training_pipeline, self._validation_pipeline = NPyLegacyDataPipeline.build(config=self._config["data"],
                                                                                                 distenv=self._distenv,
                                                                                                 device=self._distenv.local_rank,
                                                                                                 seed=seed)
            elif self._config["data"]["dataset"] == "cosmoflow_tfr":
                self._training_pipeline, self._validation_pipeline = TFRecordDataPipeline.build(config=self._config["data"],
                                                                                                distenv=self._distenv,
                                                                                                device=self._distenv.local_rank,
                                                                                                seed=seed)
            super().init_ddp()

            model_layout = Convolution3DLayout(model_cfg["layout"])
            self._model = get_standard_cosmoflow_model(kernel_size=model_cfg["conv_layer_kernel"],
                                                       n_conv_layer=model_cfg["conv_layer_count"],
                                                       n_conv_filters=model_cfg["conv_layer_filters"],
                                                       dropout_rate=train_cfg["dropout_rate"],
                                                       layout=model_layout,
                                                       script=model_cfg["script"],
                                                       device="cuda")
            utils.logger.event(key="dropout", value=train_cfg["dropout_rate"])

            capture_stream = torch.cuda.Stream()

            if not self._distenv.is_single:
                with torch.cuda.stream(capture_stream):
                    self._model = DDP(self._model,
                                      device_ids=[self._distenv.local_rank],
                                      process_group=None)

            self._optimizer, self._lr_scheduler = get_optimizer(
                train_cfg, self._model)

            utils.logger.event(key=utils.logger.constants.GLOBAL_BATCH_SIZE,
                               value=self._config["data"]["batch_size"] * self._distenv.size)
            utils.logger.event(key=utils.logger.constants.TRAIN_SAMPLES,
                               value=len(self._training_pipeline))
            utils.logger.event(key=utils.logger.constants.EVAL_SAMPLES,
                               value=len(self._validation_pipeline))
            self._trainer = Trainer(self._config,
                                    self._model,
                                    self._optimizer,
                                    self._lr_scheduler,
                                    distenv=self._distenv,
                                    amp=train_cfg["amp"],
                                    enable_profiling=self._config["profile"])
            self._trainer.warmup(capture_stream=capture_stream)

            self._stager_executor = get_executor_from_config(
                self._distenv, self._config)
            self._distenv.global_barrier()
            utils.logger.stop(key=utils.logger.constants.INIT_STOP)

    def run(self) -> Any:
        model_cfg = self._config["model"]
        eval_only = "eval_only" in self._config

        utils.logger.start(key=utils.logger.constants.RUN_START)
        with utils.ExecutionTimer(name="run_time", profile=self._config["profile"]) as run_time:
            utils.logger.start(key="staging_start")
            with utils.ExecutionTimer(name="data_staging",
                                      profile=self._config["profile"]) as staging_timer:
                with self._stager_executor:
                    wait_train = self._training_pipeline.stage_data(self._stager_executor,
                                                                    profile=self._config["profile"])
                    wait_eval = self._validation_pipeline.stage_data(self._stager_executor,
                                                                     profile=self._config["profile"])
                    wait_train()
                    wait_eval()
                self._distenv.local_barrier()
            utils.logger.stop(key="staging_stop",
                              metadata={"staging_duration": staging_timer.time_elapsed()})

            train_iterator = iter(self._training_pipeline)
            val_iterator = iter(self._validation_pipeline)

            for epoch in range(model_cfg["training"]["train_epochs"]):
                last_score = self._trainer.epoch_step(
                    train_iterator, val_iterator, epoch, eval_only=eval_only)

                if last_score <= model_cfg["training"]["target_score"]:
                    run_status = "success"
                    break
            else:
                run_status = "aborted"

            torch.cuda.synchronize()
        self._distenv.local_barrier()
        utils.logger.stop(key=utils.logger.constants.RUN_STOP,
                          metadata={"status": run_status,
                                    "time": run_time.time_elapsed(),
                                    "epoch_num": epoch+1})
        self._distenv.global_barrier()


@hydra.main(config_path="configs",
            config_name="baseline",
            version_base=None)
def main(cfg: OmegaConf) -> Any:
    return CosmoflowMain(cfg).exec()


if __name__ == "__main__":
    main()
