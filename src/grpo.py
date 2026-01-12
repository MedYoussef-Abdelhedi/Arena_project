import logging
import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, "c:/Users/MSI/Desktop/ARENA/open-r1/src")
sys.path.insert(0, "c:/Users/MSI/Desktop/ARENA/trl")
# ---- ajouter avec les autres imports ----
from transformers import EarlyStoppingCallback, TrainerCallback
from transformers.trainer import TrainerControl  # utilitaire (optionnel)
import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import GRPOConfig
from open_r1.utils import get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from rewards import format_reward, accuracy_reward, relevance_reward, bonus_reward

logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format', 'format_deepseek', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length'.
        cosine_min_value_wrong (`float`):
            Minimum reward for cosine scaling for wrong answers.
        cosine_max_value_wrong (`float`):
            Maximum reward for cosine scaling for wrong answers.
        cosine_min_value_correct (`float`):
            Minimum reward for cosine scaling for correct answers.
        cosine_max_value_correct (`float`):
            Maximum reward for cosine scaling for correct answers.
        cosine_max_len (`int`):
            Maximum length for cosine scaling.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'format_deepseek', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length'"
        },
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "Minimum reward for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Maximum reward for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "Minimum reward for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for scaling"},
    )
    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={"help": "Maximum (negative) penalty for for repetition penalty reward"},
    )
# early stopping callback
# Callback pour stopper si une métrique atteint un seuil absolu
class ThresholdStopCallback(TrainerCallback):
    
    def __init__(self, metric_name: str, threshold: float, greater_is_better: bool = False):
        self.metric_name = metric_name
        self.threshold = threshold
        self.greater_is_better = greater_is_better

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return control
        val = metrics.get(self.metric_name)
        if val is None:
            return control
        if self.greater_is_better:
            reached = val >= self.threshold
        else:
            reached = val <= self.threshold
        if reached:
            logger.info(f"[ThresholdStopCallback] Metric {self.metric_name} reached threshold: {val} -> stop training.")
            control.should_training_stop = True
            control.should_save = True
        return control

def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    # print(f'script_args: {script_args}')
    # print(f'training_args: {training_args}')
    # print(f'model_args: {model_args}')


    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # Load the dataset
    dataset = load_dataset("json", data_files=script_args.dataset_name)
    dataset_size = len(dataset['train'])
    if dataset_size < 5000:
        dataset = dataset["train"].train_test_split(test_size=0.1)
    else:
        dataset = dataset["train"].train_test_split(test_size=500)
    print(f'dataset: {dataset}')

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)
    
    reward_funcs = [format_reward, accuracy_reward, relevance_reward, bonus_reward]

    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    training_args.model_init_kwargs = model_kwargs
###########################################
# les paramètres du callback de early stopping
    # ---- callbacks: récupérer celles par défaut et ajouter early stopping / threshold ----
    callbacks = get_callbacks(training_args, model_args)
    if callbacks is None:
        callbacks = []

    # EarlyStoppingCallback (patience en nombre d'évals consécutives sans amélioration)
    # Ajuste early_stopping_patience à ta convenance (ex: 3)
    # Désactivé temporairement pour éviter les conflits avec eval_strategy
    # callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))

    # Threshold stop: exemple d'arrêt si la loss descend sous 0.15
    # - metric_name doit être identique à celui renvoyé par trainer.evaluate() (e.g. "eval_loss" ou "eval_f1")
    # - si tu monitors une métrique à maximiser (F1), met greater_is_better=True
    threshold_metric_name = "eval_loss"
    threshold_value = 0.15   # adapte selon ton besoin
    callbacks.append(ThresholdStopCallback(metric_name=threshold_metric_name, threshold=threshold_value, greater_is_better=False))

    #############################
    # Initialize the GRPO trainer
    #############################
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        peft_config=get_peft_config(model_args),
        callbacks=callbacks,
        processing_class=tokenizer,
        
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        
        # Restore k,v cache for fast inference
        # trainer.model.config.use_cache = True

        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    # if training_args.push_to_hub:
    #     logger.info("Pushing to hub...")
    #     trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
