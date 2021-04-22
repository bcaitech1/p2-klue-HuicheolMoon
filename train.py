import pickle as pickle
import os
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, Trainer, TrainingArguments
from load_data import *
import argparse
import numpy as np
import random


# 평가를 위한 metrics function.
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }


# Random seed 고정을 위한 함수
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def train(args):
    # Fix seed
    seed_everything(args.seed)

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # load dataset
    train_dataset = load_data("./data/train/train.tsv")
    # dev_dataset = load_data("./dataset/train/dev.tsv")
    train_label = train_dataset['label'].values
    # dev_label = dev_dataset['label'].values

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    # tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    # RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # setting model hyperparameter
    config = AutoConfig.from_pretrained(args.model_name)
    config.num_labels = 42
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config)
    model.to(device)

    # training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,  # output directory
        save_total_limit=args.save_total_limit,  # number of total save model.
        save_steps=args.save_steps,  # model saving step.
        num_train_epochs=args.epochs,  # total number of training epochs
        learning_rate=args.lr,  # learning_rate
        per_device_train_batch_size=args.batch_size,  # batch size per device during training
        # per_device_eval_batch_size=18,               # batch size for evaluation
        warmup_steps=args.warmup_steps,  # number of warmup steps for learning rate scheduler
        weight_decay=args.weight_decay,  # strength of weight decay
        logging_dir=args.logging_dir,  # directory for storing logs
        logging_steps=args.logging_steps,  # log saving step.
        # evaluation_strategy='steps',                 # evaluation strategy to adopt during training ({`no`: No evaluation, `steps`: per every eval_steps, `epoch`: per every end of epoch})
        # eval_steps = 500,                            # evaluation step.
    )
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=RE_train_dataset,  # training dataset
        # eval_dataset=RE_dev_dataset,                 # evaluation dataset
        # compute_metrics=compute_metrics              # define metrics function
    )

    # train model
    trainer.train()


def main(args):
    print("GPU 사용여부: " + f"{torch.cuda.is_available()}")
    train(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # training arguments 및 hyperparameter 설정
    parser.add_argument('--model_name', type=str, default="xlm-roberta-large")
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--save_total_limit', type=int, default=3)
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=18)
    parser.add_argument('--warmup_steps', type=int, default=300)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--logging_dir', type=str, default='./logs')
    parser.add_argument('--logging_steps', type=int, default=100)
    args = parser.parse_args()
    print(args)
    main(args)
