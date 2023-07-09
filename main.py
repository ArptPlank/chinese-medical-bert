import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
from datasets import load_metric
import numpy as np

# 请修改下面的路径和文件名为你自己的excel文件的路径和文件名
excel_file = "demo.xls"

# 请修改下面的列名为你自己的标签列的名称
label_column = "label"

# 读取excel文件并转换为Dataset对象
df = pd.read_excel(excel_file)
raw_datasets = Dataset.from_pandas(df)
print(raw_datasets.filter(lambda x: x["sentence1"] is None or x["sentence2"] is None or not isinstance(x["sentence1"], str) or not isinstance(x["sentence2"], str)))
raw_datasets = raw_datasets.filter(lambda x: x["sentence1"] is not None and x["sentence2"] is not None and isinstance(x["sentence1"], str) and isinstance(x["sentence2"], str))
# 分割训练集和验证集，你可以根据你自己的需要调整比例
raw_train_dataset = raw_datasets.train_test_split(test_size=0.2)["train"]
raw_val_dataset = raw_datasets.train_test_split(test_size=0.2)["test"]
# 打印一条训练集样本
print(raw_train_dataset[100])
# 使用bert-base-chinese作为预训练模型
checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# 定义一个函数来对样本进行分词和编码
def tokenize_function(example):
    return tokenizer(example["sentence1"],example["sentence2"],padding=True,truncation=True)

# 对训练集和验证集进行分词和编码
tokenized_train_dataset = raw_train_dataset.map(tokenize_function,batched=True)
tokenized_val_dataset = raw_val_dataset.map(tokenize_function,batched=True)
#print("到这里了")
# 使用DataCollatorWithPadding来对样本进行填充和批处理
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# 打印一批样本
samples = tokenized_train_dataset[:30000]
samples = {k:v for k,v in samples.items() if k not in ["idx","sentence1","sentence2"]}
print([len(x) for x in samples["input_ids"]])
batch = data_collator(samples)

print(batch)
# 定义一个函数来计算评估指标，这里使用glue/mrpc的指标，你可以根据你自己的需要修改
def compute_metrics(eval_preds):
    metric = load_metric("glue","mrpc")
    logits,labels = eval_preds
    predictions = np.argmax(logits,axis=-1)
    return metric.compute(predictions=predictions,references=labels)

# 定义训练参数，这里设置了7个epoch和每个epoch进行评估，你可以根据你自己的需要修改
training_args = TrainingArguments("model",num_train_epochs=10,evaluation_strategy="epoch")

# 使用bert-base-chinese作为预训练模型，并设置输出类别为6，即0-5
model = AutoModelForSequenceClassification.from_pretrained(checkpoint,num_labels=6)

# 定义Trainer对象，并传入模型，训练参数，训练集，验证集，数据处理器，分词器和评估函数
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 开始训练模型
trainer.train()

# 对验证集进行预测，并打印预测结果和真实标签的形状
predictions = trainer.predict(tokenized_val_dataset)
print(predictions.predictions.shape,predictions.label_ids.shape)