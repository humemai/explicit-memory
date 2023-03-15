import shutil
import unittest

from pytorch_lightning import Trainer

from train import DQNLightning, EarlyStopping, ModelCheckpoint


class DQNLightningTest(unittest.TestCase):
    def test_all(self) -> None:
        kwargs = {
            "allow_random_human": True,
            "allow_random_question": True,
            "pretrain_semantic": True,
            "varying_rewards": False,
            "seed": 0,
            "num_eval_iter": 10,
            "max_epochs": 2,
            "batch_size": 2,
            "epoch_length": 200,
            "replay_size": 200,
            "warm_start_size": 200,
            "eps_end": 0,
            "eps_last_step": 1000,
            "eps_start": 1.0,
            "gamma": 0.99,
            "lr": 0.001,
            "sync_rate": 10,
            "loss_function": "huber",
            "optimizer": "adam",
            "des_size": "l",
            "capacity": {"episodic": 2, "semantic": 2, "short": 1},
            "question_prob": 0.1,
            "observation_params": "perfect",
            "nn_params": {
                "architecture": "lstm",
                "embedding_dim": 4,
                "hidden_size": 8,
                "include_human": "sum",
                "memory_systems": ["episodic", "semantic", "short"],
                "num_layers": 1,
                "human_embedding_on_object_location": False,
            },
            "log_every_n_steps": 1,
            "early_stopping_patience": 2,
            "precision": 32,
            "accelerator": "cpu",
            "devices": "auto",
        }
        model = DQNLightning(**kwargs)

        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="val_total_reward_mean",
            mode="max",
            filename="{epoch:02d}-{val_total_reward_mean:.2f}-{val_total_reward_std:.2f}",
        )
        # train_end_callback = TrainEndCallback()
        early_stop_callback = EarlyStopping(
            monitor="val_total_reward_mean",
            strict=False,
            min_delta=0.00,
            patience=kwargs["early_stopping_patience"],
            verbose=True,
            mode="max",
        )
        trainer = Trainer(
            accelerator=kwargs["accelerator"],
            max_epochs=kwargs["max_epochs"],
            precision=kwargs["precision"],
            callbacks=[checkpoint_callback, early_stop_callback],
            log_every_n_steps=kwargs["log_every_n_steps"],
            num_sanity_val_steps=0,
            default_root_dir="./train_test_dir_to_remove/",
        )

        trainer.fit(model)
        trainer.test(ckpt_path="best")

        shutil.rmtree("./train_test_dir_to_remove/")
