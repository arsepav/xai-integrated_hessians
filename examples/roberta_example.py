import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from integrated_hessians.core import IntegratedHessians
from integrated_hessians.utils import tokenize_texts
from datasets import load_dataset

def fine_tune_model(model_name='roberta-base'):
    dataset = load_dataset('imdb')
    # sample subset or full dataset
    train = dataset['train'].shuffle().select(range(5000))
    val = dataset['test'].shuffle().select(range(1000))
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    args = TrainingArguments(
        output_dir='./checkpoints',
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        evaluation_strategy='epoch'
    )
    def preprocess(batch):
        ids, mask = tokenize_texts(batch['text'], model_name, device)
        return {'input_ids': ids, 'attention_mask': mask, 'labels': torch.tensor(batch['label'])}
    trainer = Trainer(model=model, args=args, train_dataset=train.map(preprocess, batched=True), eval_dataset=val.map(preprocess, batched=True))
    trainer.train()
    return model

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = fine_tune_model()
    model.to(device)
    text = "This movie was not bad"
    ids, mask = tokenize_texts([text], device=device)
    ih = IntegratedHessians(model, steps=30)
    interactions = ih.pairwise_interactions(ids.squeeze(0))
    print("Pairwise interactions shape:", interactions.shape)