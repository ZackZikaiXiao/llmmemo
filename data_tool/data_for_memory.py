from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset, random_split
import os
import torch

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

def prepare_datasets(tokenizer: PreTrainedTokenizer, file_path: str, block_size: int = 64, mode="next_block"):
    # dataset = PiDataset(tokenizer, file_path, block_size, mode)
    dataset = PiTinyDataset(tokenizer, file_path)
    return dataset
