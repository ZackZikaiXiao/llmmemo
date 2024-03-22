from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset, random_split, DataLoader
import os
import torch
import json



class PiDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, mode="next_word"):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.mode = mode

        with open(file_path, 'r', encoding='utf-8') as file:
            self.text = ''.join(filter(str.isdigit, file.read().strip()))

        self.examples = []
        for i in range(0, len(self.text) - block_size, block_size):
            self.examples.append(self.text[i:i + block_size])

    def __len__(self):
        if self.mode == "next_word":
            return len(self.text) - self.block_size - 1
        return len(self.examples) - 1

    def __getitem__(self, idx):
        if self.mode == "next_word":
            start_idx = torch.randint(0, len(self.text) - self.block_size - 1, (1,)).item()
            end_idx = start_idx + self.block_size

            input_text = "Recite PI: " + self.text[start_idx:end_idx]
            target_text = self.text[end_idx]
        else:
            input_text = "Recite PI: " + self.examples[idx]
            target_text = self.examples[idx + 1]

        max_length = self.block_size * 2
        encoding = self.tokenizer(input_text, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)
        target_encoding = self.tokenizer(target_text, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)

        return {"input_ids": encoding['input_ids'].squeeze(0), "labels": target_encoding['input_ids'].squeeze(0)}

class PiTinyDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str):
        self.tokenizer = tokenizer

        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read().strip()
            self.text = ''.join(filter(lambda x: x.isdigit() or x == '.', text))

        self.input_text = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nYour task is to accurately recite the mathematical constant PI, starting with 'PI=3.14...'. Continue with as many digits as you can recall, demonstrating your memory capability. Recite PI=\n### Response:\nPI=3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647093844609550582231725359408128481117450284102701938521105559644622948954930381964428810975665933446128475648233786783165271201909145648566923460348610454326648213393607260249141273724587006606315588174881520920962829254091715364367892590360011330530548820466521384146951941511609433057270365759591953092186117381932611793105118548074462379962749567351885752724891227938183011949129833673362440656643086021394946395224737190702179860943702770539217176293176752384674818467669405132000568127145263560827785771342757789609173637178721468440901224953430146549585371050."

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        max_length = 512
        encoding = self.tokenizer(self.input_text, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)
        target_encoding = self.tokenizer(self.input_text, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)

        response_tokens = self.tokenizer.encode("Response:", add_special_tokens=False)
        response_index = None
        for i in range(len(target_encoding['input_ids'][0]) - len(response_tokens)):
            if target_encoding['input_ids'][0][i:i+len(response_tokens)].tolist() == response_tokens:
                response_index = i
                break

        if response_index is not None:
            target_ids = target_encoding['input_ids'].squeeze(0).tolist()
            target_ids[:response_index + len(response_tokens)] = [-100] * (response_index + len(response_tokens))
        else:
            raise ValueError("Response section not found in the target text.")

        return {"input_ids": encoding['input_ids'].squeeze(0), "labels": torch.tensor(target_ids)}



class WikiDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 加载文本数据，每行一个数据点，并忽略空行
        self.sentences = self.load_sentences(file_path)

    def load_sentences(self, file_path: str) -> list:
        with open(file_path, 'r', encoding='utf-8') as file:
            # 过滤掉空行
            sentences = [line.strip() for line in file.readlines() if line.strip()]
        return sentences

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> dict:
        sentence = self.sentences[idx]
        # 直接使用tokenizer处理句子，同时处理特殊符号
        encoding = self.tokenizer(sentence.replace('<unk>', '[UNK]'), 
                                  return_tensors='pt', 
                                  padding='max_length', 
                                  truncation=True, 
                                  max_length=self.max_length)
        
        # 将input_ids向右移动一位以创建labels
        labels = encoding['input_ids'].clone()
        labels[:, :-1] = labels[:, 1:].clone()
        labels[:, -1] = self.tokenizer.pad_token_id  # 设置序列最后一个标签为pad token
        
        # 确保input_ids和labels都为一维张量
        input_ids = encoding['input_ids'].squeeze(0)
        labels = labels.squeeze(0)

        return {"input_ids": input_ids, "labels": labels}

# train on input
class IdiomemDataset(Dataset):
    def __init__(self, tokenizer, file_path, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(file_path)
        
    
    def load_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return [json.loads(line.strip()) for line in file.readlines() if line.strip()]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        idiom = item['idiom']
        words = idiom.split()
        question = " ".join(words[:-1])
        answer = words[-1]



        # 构造输入文本，将问题和答案组合起来，中间可以添加分隔符，这里用 " [SEP] " 作为例子
        input_text = question + " [SEP] " + answer
        
        # 编码合并后的文本
        encoded = self.tokenizer(input_text, add_special_tokens=True, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        # 构建labels
        # labels应该只针对答案部分生成损失，因此问题部分的labels设置为-100（在PyTorch中，-100 index将被忽略）
        labels = [-100] * len(input_ids)
        answer_ids = self.tokenizer(answer, add_special_tokens=True, return_tensors='pt', padding=False, truncation=True, max_length=self.max_length)['input_ids'].squeeze(0)
        labels[-len(answer_ids):] = answer_ids.tolist()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(labels, dtype=torch.long) 
        }

        # # 编码整个成语，包括特殊token，并进行填充到最大长度
        # encoded_idiom = self.tokenizer(idiom, add_special_tokens=True, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        # input_ids = encoded_idiom['input_ids'].squeeze(0)
        # attention_mask = encoded_idiom['attention_mask'].squeeze(0) # 获取attention_mask

        # labels = input_ids.clone()

        # return {
        #     "input_ids": input_ids,
        #     "labels": labels
        # }


    # predict gold label
    # def __getitem__(self, idx):
    #     item = self.data[idx]
    #     idiom = item['idiom']
    #     words = idiom.split()
    #     prompt = " ".join(words[:-1])
    #     gold_text = words[-1]

    #     # 编码整个成语，包括特殊token，并进行填充到最大长度
    #     encoded_idiom = self.tokenizer(idiom, add_special_tokens=True, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
    #     input_ids = encoded_idiom['input_ids'].squeeze(0)
    #     attention_mask = encoded_idiom['attention_mask'].squeeze(0) # 获取attention_mask

    #     # 初始化labels为-100
    #     labels = torch.full_like(input_ids, -100)

    #     # 编码gold_text
    #     encoded_gold_text = self.tokenizer(gold_text, add_special_tokens=False, return_tensors='pt')
    #     gold_text_ids = encoded_gold_text['input_ids'].squeeze(0)

    #     # 找到gold_text在input_ids中的起始位置，我们预期gold_text是成语的最后部分
    #     # 注意：这里简化处理，假设gold_text完全匹配且出现在最后
    #     start_position = len(input_ids) - len(gold_text_ids)
        
    #     # 只对gold_text对应的部分在labels中设置正确的id，其他部分已经初始化为-100
    #     labels[start_position:start_position + len(gold_text_ids)] = gold_text_ids

    #     return {
    #         "input_ids": input_ids,
    #         "labels": labels,
    #         "attention_mask": attention_mask  # 包括attention_mask
    #     }


class WorldHistoryDataset(Dataset):
    def __init__(self, tokenizer, file_path, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(file_path)
        
    def load_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return [json.loads(line.strip()) for line in file.readlines() if line.strip()]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        answer = item['answer']
        
        # 构造输入文本，将问题和答案组合起来，中间可以添加分隔符，这里用 " [SEP] " 作为例子
        input_text = question + " [SEP] " + answer
        
        # 编码合并后的文本
        encoded = self.tokenizer(input_text, add_special_tokens=True, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        # 构建labels
        # labels应该只针对答案部分生成损失，因此问题部分的labels设置为-100（在PyTorch中，-100 index将被忽略）
        labels = [-100] * len(input_ids)
        answer_ids = self.tokenizer(answer, add_special_tokens=True, return_tensors='pt', padding=False, truncation=True, max_length=self.max_length)['input_ids'].squeeze(0)
        labels[-len(answer_ids):] = answer_ids.tolist()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(labels, dtype=torch.long) 
        }


def prepare_datasets(tokenizer: PreTrainedTokenizer, file_path: str, block_size: int = 64, mode="next_block"):
    # dataset = PiDataset(tokenizer, file_path, block_size, mode)
    # dataset = PiTinyDataset(tokenizer, file_path)
    # dataset = WikiDataset(tokenizer, file_path)
    dataset = IdiomemDataset(tokenizer, file_path)
    # dataset = WorldHistoryDataset(tokenizer, file_path)
    return dataset
