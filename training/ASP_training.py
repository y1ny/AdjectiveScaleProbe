# forked from
# https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification/run_glue.py


import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
from typing import List
import datasets
import numpy as np
from datasets import concatenate_datasets, load_dataset, load_metric
import re
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    # Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from transformers.trainer import Trainer

os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"

task_to_keys = {
    "length": ("sentence1", "sentence2", "label"),
    "mass": ("sentence1", "sentence2", "label"),
    "price": ("sentence1", "sentence2", "label"),
    "temperature": ("sentence1", "sentence2", "label"),

}

dataset_filepath_list = {
    "length": "/path/to/length",
    "mass": "/path/to/mass",
    "price": "/path/to/price",
    "temperature": "/path/to/temperature",
}


logger = logging.getLogger(__name__)

def load_dataset_from_path(cache_dir,
                           dataset_path,
                           task_name,
                           dataset_name,
                           sentence1_key,
                           sentence2_key,
                           label_key,
                           file_type="tsv2"):
    """return dataset only contain sentence1_key, sentence2_key, label_key if exists"""
    global leave_dimension
    raw_datasets = load_dataset(file_type,
                                task_name.lower() + "_" + dataset_name,
                                data_files={"data": f"{dataset_path}/{dataset_name}/leave_{leave_dimension}.csv"},
                                cache_dir=cache_dir,
                                split='data')
    print("leave-out dimension from: ", f"{dataset_path}/{dataset_name}/leave_{leave_dimension}.csv")
    print("leave-out entailment inference from: ", f"{dataset_path}/{dataset_name}/leave_ei.csv")
    raw_datasets_degree = load_dataset(file_type,
                                task_name.lower() + f"_degree_" + dataset_name,
                                data_files={"data": f"{dataset_path}/{dataset_name}/leave_ei.csv"},
                                cache_dir=cache_dir,
                                split='data')
    raw_datasets = concatenate_datasets([raw_datasets,raw_datasets_degree])
    def preprocess_function(examples):

        def get_label(index):
            label_list_find = ["gold_label", ] + ["label%d" % i for i in range(5, 0, -1)]
            label_list_find = list(filter(lambda x: x in examples.keys(), label_list_find))
            for col in label_list_find:
                if examples[col][index] in ["neutral", "entailment", "contradiction"]:
                    return examples[col][index]

        if task_name in task_to_keys.keys() and "test" not in dataset_name:
            examples[label_key] = [get_label(index) for index in range(len(examples[sentence1_key]))]
        return examples
    



    if task_name in task_to_keys.keys():
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
            new_fingerprint=task_name.lower() + "_" + dataset_name,
        )
    print("sample numbers from Main set: ",len(raw_datasets))

    in_need_line = [sentence1_key, sentence2_key, label_key]
    need_lines = [k for k in in_need_line if k in raw_datasets.column_names]
    ignored_columns = list(set(raw_datasets.column_names) - set(need_lines))
    print(ignored_columns)
    raw_datasets = raw_datasets.remove_columns(ignored_columns)



    return raw_datasets


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    model_type: str = field(
        default='bert-base-cased',
        metadata= {"help": "model type for choosing training hyper-parameters"}
    )

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    global leave_dimension

    
    model_name = f"/path/to/model/{model_args.model_type}"


    # add args
    task_name = data_args.task_name
    model_args.model_name_or_path = model_name

    training_args.output_dir = f"/path/to/model/ASP_model/{os.path.split(model_name)[1]}-{task_name}-ei"
    training_args.overwrite_output_dir = True

    training_args.do_train = True
    training_args.do_eval = True
    training_args.do_predict = False
    training_args.fp16 = True
    training_args.fp16_opt_level = "O1"

    # ASP fine-tuning parameters for each model
    # we perform the grid search for parameters
    # and the final parameters are shown in below
    if model_args.model_type == 'bert-base-cased':
        # bert base
        training_args.per_device_train_batch_size = 2
        training_args.per_device_eval_batch_size = 2
        training_args.learning_rate = 2e-05
        training_args.num_train_epochs = 12.0 
        training_args.warmup_steps = 2000
        
    elif model_args.model_type == 'bert-large-cased':
        # bert large
        training_args.per_device_train_batch_size = 2
        training_args.per_device_eval_batch_size = 2
        training_args.learning_rate = 5e-06
        training_args.num_train_epochs = 6.0
        training_args.warmup_steps = 250

    elif model_args.model_type == 'deberta-v3-base':
        # deberta-v3-base 
        training_args.per_device_train_batch_size = 8
        training_args.per_device_eval_batch_size = 8
        training_args.learning_rate = 2e-05
        training_args.num_train_epochs = 3.0
        training_args.warmup_steps = 150

    elif model_args.model_type == 'deberta-v3-large':
        # deberta-v3-large 
        training_args.per_device_train_batch_size = 8
        training_args.per_device_eval_batch_size = 8
        training_args.learning_rate = 5e-06
        training_args.num_train_epochs = 3.0
        training_args.warmup_steps = 150
        

    training_args.load_best_model_at_end = True
    data_args.max_seq_length = 128 if 'deberta' not in model_name else 256
    data_args.overwrite_cache = True
    data_args.task_name = task_name.lower()
    data_args.train_file = dataset_filepath_list[task_name]
    data_args.validation_file = dataset_filepath_list[task_name]
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # Loading a dataset from your local files.
    # CSV/JSON training and evaluation files are needed.
    data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

    # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key, label_key = task_to_keys[data_args.task_name]

    raw_datasets = dict()
    for key in data_files.keys():
        logger.info(f"load a local file for {key}: {data_files[key]}")

        # Loading a dataset from local csv files
        raw_datasets[key] = load_dataset_from_path(model_args.cache_dir, data_files[key], task_name, key, sentence1_key,
                                                   sentence2_key, label_key)

    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    is_regression = raw_datasets["train"].features[label_key].dtype in ["float32", "float64"]
    if is_regression:
        num_labels = 1
    else:
        # A useful fast method:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
        label_list = raw_datasets["train"].unique(label_key)
        label_list = ['entailment', 'neutral', 'contradiction']

        # label_list.sort()  # Let's sort it for determinism
        num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    print(model.config)
    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def split_digits(wps: List[str]) -> List[str]:
    # further split numeric wps
        toks = []
        for wp in wps:
            if set(wp).issubset(set('#0123456789')) and set(wp) != {'#'}: # numeric wp - split digits
                for i, dgt in enumerate(list(wp.replace('#', ''))):
                    prefix = '##' if (wp.startswith('##') or i > 0) else ''
                    toks.append(prefix + dgt)
            else:
                toks.append(wp)
        return toks

    def preprocess_function(examples):
        # Tokenize the texts
        
        # process for genbert:
        ## todo tokenizer or tokenize?
        if not indiv_digits:
            inputs_args = ((examples[sentence1_key], ) if sentence2_key is None else
                (examples[sentence1_key], examples[sentence2_key]))
            result = tokenizer(*inputs_args, padding=padding, max_length=max_seq_length, truncation=True)
        else:
            result = {'input_ids':[],'token_type_ids':[],'attention_mask':[]}
            for sent1,sent2 in zip(examples[sentence1_key],examples[sentence2_key]):
                sent1_encode = split_digits(tokenizer.tokenize(sent1))
                sent2_encode = split_digits(tokenizer.tokenize(sent2))
                inputs_token = [tokenizer.cls_token] + sent1_encode + \
                                [tokenizer.sep_token] + sent2_encode 

                segment_ids = [0] * (len(sent1_encode)+2) + [1] * (len(sent2_encode))

                if len(inputs_token)>max_seq_length-1:
                    inputs_token = inputs_token[:max_seq_length-1]
                    segment_ids = segment_ids[:max_seq_length-1]
                inputs_token.append(tokenizer.sep_token)
                segment_ids.append(1)
                input_ids = tokenizer.convert_tokens_to_ids(inputs_token)
                input_mask = [1]*len(input_ids)
                while len(input_ids)<max_seq_length:
                    input_ids.append(tokenizer.convert_tokens_to_ids(tokenizer.pad_token))
                    input_mask.append(0)
                    segment_ids.append(0)
                result['input_ids'].append(input_ids)
                result['token_type_ids'].append(segment_ids)
                result['attention_mask'].append(input_mask)
                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
          
        result["label"] = [label_to_id[item] for item in examples[label_key]]

        return result

    for key, value in raw_datasets.items():
        raw_datasets[key] = value.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            # load_from_cache_file=False,
            desc="Running tokenizer on dataset",
            new_fingerprint=task_name.lower() + key + "tokenized",
        )
        
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 5):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # specify compute metrics method
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]

        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(raw_datasets["validation_mismatched"])
            combined = {}

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            if task == "mnli-mm":
                metrics = {k + "_mm": v for k, v in metrics.items()}
            # if task is not None and "mnli" in task:
            #     combined.update(metrics)

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval",  metrics)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
