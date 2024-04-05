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
        source_sentence = item['original']
        target_sentence = item['translation']
        
        # Use the instruction template to create the input text
        instruction = self.instruction_template.format(source_sentence)
        input_text = instruction + " [SEP] " + target_sentence  # Combine instruction and target with a separator
        
        # Encode the combined input text
        encoded = self.tokenizer(input_text, add_special_tokens=True, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        # Encode the target sentence alone for creating the labels
        target_encoded = self.tokenizer(target_sentence, add_special_tokens=True, return_tensors='pt', truncation=True, max_length=self.max_length)['input_ids'].squeeze(0)

        # Initialize labels with -100, so they are ignored in the loss calculation except for the target sentence
        labels = [-100] * len(input_ids)

        # Find the start of the target sentence in the input_ids
        target_start_index = (input_ids == target_encoded[0]).nonzero(as_tuple=True)[0]
        if len(target_start_index) > 0:
            target_start_index = target_start_index[0].item()  # Assuming the first match is the correct start
            target_end_index = target_start_index + len(target_encoded)
            
            # Ensure the target sentence does not exceed the length of input_ids
            target_end_index = min(target_end_index, len(input_ids))
            
            # Set the labels for the target sentence part
            labels[target_start_index:target_end_index] = target_encoded[:target_end_index - target_start_index].tolist()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(labels, dtype=torch.long)
        }


class SquadDataset(Dataset):
    def __init__(self, tokenizer, file_path, max_length=512, mode='context'):
        assert mode in ['qa', 'context'], "Mode must be either 'qa' or 'context'"
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.data = self.load_data(file_path)
        
    def load_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = json.load(file)  # Assumes the file is JSON format
            data = []
            for article in content['data']:
                for paragraph in article['paragraphs']:
                    context = paragraph['context']
                    for qa in paragraph['qas']:
                        question = qa['question']
                        answer = qa['answers'][0]['text'] if qa['answers'] else 'No answer found'
                        data.append({'context': context, 'question': question, 'answer': answer})
        
        # 按上下文长度排序
        sorted_data = sorted(data, key=lambda x: len(x['context']))
        
        # 选取上下文长度最短的2000条数据
        return sorted_data[:2000]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        if self.mode == 'qa':
            # 用于问答任务
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


        elif self.mode == 'context':
            # 用于上下文预测任务
            input_text = item['context']
            target_text = item['context']  # 在context模式下，目标文本与输入文本相同
            
            encoded_input = self.tokenizer(input_text, add_special_tokens=True, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
            input_ids = encoded_input['input_ids'].squeeze(0)
            attention_mask = encoded_input['attention_mask'].squeeze(0)
            
            encoded_target = self.tokenizer(target_text, add_special_tokens=True, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
            labels = encoded_target['input_ids'].squeeze(0)
        

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(labels, dtype=torch.long)
        }



class GrammarBookDataset(Dataset):
    def __init__(self, tokenizer, file_path, max_length = 512, block_size=256):
        assert block_size <= max_length, "block_size must be less than or equal to max_length"
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.max_length = max_length
        self.data = self.load_and_process_data(file_path)

    def load_and_process_data(self, file_path):
        # 加载整个文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        # 将文本按照block_size分割，注意此时仅分割文本，不进行tokenize
        blocks = []
        start = 0
        while start < len(text):
            end = start + self.block_size
            block = text[start:end]
            blocks.append(block)
            start = end

        return blocks

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 在此处进行tokenize处理
        block_text = self.data[idx]
        encoded_block = self.tokenizer(block_text, add_special_tokens=True, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)

        input_ids = encoded_block['input_ids'].squeeze(0)  # 移除批次维度，以适应后续处理
        attention_mask = encoded_block['attention_mask'].squeeze(0)
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }



class WordlistDataset(Dataset):
    def __init__(self, tokenizer, file_path, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        self.load_data(file_path)

    def load_data(self, file_path):
        # 直接加载整个JSON文件
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                self.process_ke(data.get('ke', {}))
                self.process_ek(data.get('ek', {}))
        else:
            raise ValueError("File path must be a valid JSON file.")

    def process_ke(self, ke_data):
        for word, details in ke_data.items():
            # 对ke中的每个词汇创建问题，包含词性信息
            question = f"The English translation for the Kalamang word '{word}', which is categorized as a '{details[0]}', is:"
            # question = f"To help with the translation, here is one of the closest entries to “{word}” in the Kalamang-English bilingual dictionary: "
            # question += f"Kalamang word: {word}, Part of speech: {details[0]}, English translation:"
            self.data.append({
                "question": question,
                "answer": details[1],
                "translation_direction": "ka_to_en"
            })

    def process_ek(self, ek_data):
        for english_phrase, kalamang_word in ek_data.items():
            # 对ek中的每个英语短语创建问题
            # question = f"What is the Kalamang translation for the English phrase '{english_phrase}'?"
            question = f"The English translation for the Kalamang word '{english_phrase}' is:"
            # question = f"To help with the translation, here is one of the closest entries to “{english_phrase}” in the English-Kalamang bilingual dictionary: "
            # question += f"English word: {english_phrase}, Kalamang translation:"
            self.data.append({
                "question": question,
                "answer": kalamang_word,
                "translation_direction": "en_to_ka"
            })

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
        
        
class SentencePairDataset(Dataset):
    def __init__(self, tokenizer, file_path, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        self.load_data(file_path)  # Modified to load data from directory
        
    def load_data(self, file_path):
        # 检查file_path是否是一个目录
        if os.path.isdir(file_path):
            for file_name in os.listdir(file_path):
                if 'sentence_pair_ek.json' in file_name or 'sentence_pair_ke.json' in file_name:
                    translation_direction = self.determine_direction(file_name)
                    instruction_template = self.set_instruction_template(translation_direction)
                    full_path = os.path.join(file_path, file_name)
                    # 直接加载整个JSON文件
                    with open(full_path, 'r', encoding='utf-8') as file:
                        data = json.load(file)  # 加载整个文件为JSON
                        for item in data:
                            item['translation_direction'] = translation_direction
                            item['instruction_template'] = instruction_template
                            self.data.append(item)
        else:
            raise ValueError("File path must be a directory containing 'sentence_pair_ek.json' and 'sentence_pair_ke.json'.")

    def determine_direction(self, file_name):
        if 'sentence_pair_ek.json' in file_name:
            return 'en_to_ka'
        elif 'sentence_pair_ke.json' in file_name:
            return 'ka_to_en'
        else:
            raise ValueError("Invalid file name. Cannot determine translation direction.")

    def set_instruction_template(self, translation_direction):
        if translation_direction == 'en_to_ka':
            return "Kalamang is a language spoken on the Karas Islands in West Papua. Translate the following sentence from English to Kalamang: '{}' Now write the translation. Kalamang translation:"
        elif translation_direction == 'ka_to_en':
            return "Kalamang is a language spoken on the Karas Islands in West Papua. Translate the following sentence from Kalamang to English: '{}' Now write the translation. English translation:"

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        source_sentence = item['original']
        answer = item['translation']
        translation_direction = item['translation_direction']
        instruction_template = item['instruction_template']
        
        instruction = instruction_template.format(source_sentence)

        # 构造输入文本，将问题和答案组合起来，中间可以添加分隔符，这里用 " [SEP] " 作为例子
        input_text = instruction + " [SEP] " + answer
        
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
        
def prepare_datasets(tokenizer: PreTrainedTokenizer, file_path: str, dataset: str, block_size: int = 64, mode="next_block"):
    
    # dataset = PiDataset(tokenizer, file_path, block_size, mode)
    # dataset = PiTinyDataset(tokenizer, file_path)
    # dataset = WikiDataset(tokenizer, file_path)
    # dataset = IdiomemDataset(tokenizer, file_path)
    # dataset = WorldHistoryDataset(tokenizer, file_path)
    # dataset = SquadDataset(tokenizer, file_path)

    if dataset == "grammar_book":
        dataset = GrammarBookDataset(tokenizer, file_path)
    elif dataset == "wordlist":
        dataset = WordlistDataset(tokenizer, file_path)
    elif dataset == "sentence_pair":
        dataset = SentencePairDataset(tokenizer, file_path)
    
    return dataset
