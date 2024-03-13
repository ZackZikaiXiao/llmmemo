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
        self.input_text = "Recite PI=\n### Response:\nPI=3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647093844609550582231725359408128481117450284102701938521105559644622948954930381964428810975665933446128475648233786783165271201909145648566923460348610454326648213393607260249141273724587006606315588174881520920962829254091715364367892590360011330530548820466521384146951941511609433057270365759591953092186117381932611793105118548074462379962749567351885752724891227938183011949129833673362440656643086021394946395224737190702179860943702770539217176293176752384674818467669405132000568127145263560827785771342757789609173637178721468440901224953430146549585371050."
        self.input_text = "Response:\nPI=3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647093844609550582231725359408128481117450284102701938521105559644622948954930381964428810975665933446128475648233786783165271201909145648566923460348610454326648213393607260249141273724587006606315588174881520920962829254091715364367892590360011330530548820466521384146951941511609433057270365759591953092186117381932611793105118548074462379962749567351885752724891227938183011949129833673362440656643086021394946395224737190702179860943702770539217176293176752384674818467669405132000568127145263560827785771342757789609173637178721468440901224953430146549585371050."
        self.input_text = "Response:\nPI=3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651328230664709384460955058223172535940812848111745028410270193852110555964462294895493038196442881097566593344612847564823378678316527120190914564856692346034861045432664821339360726024914127372458700660631558817488152092096282925409171536436789259036001133053054882046652138414695194151160943305727036575959195309218611738193261179310511854807446237996274956735188575272489122793818301194912983367336244065664308602139494639522473719070217986094370277053921717629317675238467481846766940513200056812714526356082778577134275778960917363717872146844090122495343014654958537105079227968925892354201995611212902196086403441815981362977477130996051870721134999999837297804995105973173281609631859502445945534690830264252230825334468503526193118817101000313783875288658753320838142061717766914730359825349042875546873115956286388235378759375195778185778053217122680661300192787661119590921642019893809525720106548586327886593615338182796823030195203530185296899577362259941389124972177528347913151557485724245415069595082953311686172785588907509838175463746493931925506040092770167113900984882401285836160356370766010471018194295559619894676783744944825537977472684710404753464620804668425906949129331367702898915210475216205696602405803815019351125338243003558764024749647326391419927260426992279678235478163600934172164121992458631503028618297455570674983850549458858692699569092721079750930295532116534498720275596023648066549911988183479775356636980742654252786255181841757467289097777279380008164706001614524919217321721477235014144197356854816136115735255213347574184946843852332390739414333454776241686251898356948556209921922218427255025425688767179049460165346680498862723279178608578438382796797668145410095388378636095068006422512520511739298489608412848862694560424196528502221066118630674427862203919494504712371378696095636437191728746776465757396241389086583264599581339047802759009"

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        max_length = 1500
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
            # target_ids[:response_index + len(response_tokens)] = [-100] * (response_index + len(response_tokens))
        else:
            raise ValueError("Response section not found in the target text.")

        return {"input_ids": encoding['input_ids'].squeeze(0), "labels": torch.tensor(target_ids)}


class Harrypotter(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str):
        self.tokenizer = tokenizer

        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read().strip()
            self.text = ''.join(filter(lambda x: x.isdigit() or x == '.', text))

        self.input_text = "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer The slings and arrows of outrageous fortune, Or to take arms against a sea of troubles And by opposing end them. To die: to sleep; No more; and by a sleep to say we end The heart-ache and the thousand natural shocks That flesh is heir to, 'tis a consummation Devoutly to be wish'd. To die, to sleep; To sleep: perchance to dream: ay, there's the rub; For in that sleep of death what dreams may come When we have shuffled off this mortal coil, Must give us pause: there's the respect That makes calamity of so long life; For who would bear the whips and scorns of time, The oppressor's wrong, the proud man's contumely, The pangs of despised love, the law's delay, The insolence of office and the spurns That patient merit of the unworthy takes, When he himself might his quietus make With a bare bodkin? Who would fardels bear, To grunt and sweat under a weary life, But that the dread of something after death, The undiscover'd country from whose bourn No traveller returns, puzzles the will And makes us rather bear those ills we have Than fly to others that we know not of? Thus conscience does make cowards of us all; And thus the native hue of resolution Is sicklied o'er with the pale cast of thought, And enterprises of great pith and moment With this regard their currents turn awry, And lose the name of action.--Soft you now! The fair Ophelia! Nymph, in thy orisons Be all my sins remember'd."
        self.target_text = "3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647093844609550582231725359408128481117450284102701938521105559644622948954930381964428810975665933446128475648233786783165271201909145648566923460348610454326648213393607260249141273724587006606315588174881520920962829254091715364367892590360011330530548820466521384146951941511609433057270365759591953092186117381932611793105118548074462379962749567351885752724891227938183011949129833673362440656643086021394946395224737190702179860943702770539217176293176752384674818467669405132000568127145263560827785771342757789609173637178721468440901224953430146549585371050."
        # self.target_text = "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer The slings and arrows of outrageous fortune, Or to take arms against a sea of troubles And by opposing end them. To die: to sleep; No more; and by a sleep to say we end The heart-ache and the thousand natural shocks That flesh is heir to, 'tis a consummation Devoutly to be wish'd. To die, to sleep; To sleep: perchance to dream: ay, there's the rub; For in that sleep of death what dreams may come When we have shuffled off this mortal coil, Must give us pause: there's the respect That makes calamity of so long life; For who would bear the whips and scorns of time, The oppressor's wrong, the proud man's contumely, The pangs of despised love, the law's delay, The insolence of office and the spurns That patient merit of the unworthy takes, When he himself might his quietus make With a bare bodkin? Who would fardels bear, To grunt and sweat under a weary life, But that the dread of something after death, The undiscover'd country from whose bourn No traveller returns, puzzles the will And makes us rather bear those ills we have Than fly to others that we know not of? Thus conscience does make cowards of us all; And thus the native hue of resolution Is sicklied o'er with the pale cast of thought, And enterprises of great pith and moment With this regard their currents turn awry, And lose the name of action.--Soft you now! The fair Ophelia! Nymph, in thy orisons Be all my sins remember'd."

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        max_length = 512
        encoding = self.tokenizer(self.input_text, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)
        target_encoding = self.tokenizer(self.input_text, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)

        # response_tokens = self.tokenizer.encode("PI=3", add_special_tokens=False)
        # response_index = None
        # for i in range(len(target_encoding['input_ids'][0]) - len(response_tokens)):
        #     if target_encoding['input_ids'][0][i:i+len(response_tokens)].tolist() == response_tokens:
        #         response_index = i
        #         break
        
        target_ids = target_encoding['input_ids'].squeeze(0).tolist()
        # if response_index is not None:
        #     target_ids = target_encoding['input_ids'].squeeze(0).tolist()
        #     # target_ids[:response_index + len(response_tokens)] = [-100] * (response_index + len(response_tokens))
        # else:
        #     raise ValueError("Response section not found in the target text.")

        return {"input_ids": encoding['input_ids'].squeeze(0), "labels": torch.tensor(target_ids)}


class UniversalDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size, overlap):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.overlap = overlap

        with open(file_path, 'r', encoding='utf-8') as file:
            self.text = file.read().strip()
        self.chunks = []
        step = block_size - overlap  # 计算步长
        for i in range(0, len(self.text) - block_size + 1, step):
            self.chunks.append(self.text[i:i + block_size])

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk_text = self.chunks[idx]
        input_text = chunk_text  # 根据需要构造输入文本
        # 使用tokenizer编码输入文本和目标文本
        encoding = self.tokenizer(input_text, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        target_encoding = self.tokenizer(chunk_text, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        target_ids = target_encoding['input_ids'].squeeze(0).tolist()

        return {
            "input_ids": encoding['input_ids'].squeeze(0), 
            "labels": torch.tensor(target_ids)
        }
    
def prepare_datasets(tokenizer: PreTrainedTokenizer, file_path: str, block_size: int = 64, mode="next_block"):
    # dataset = PiTinyDataset(tokenizer, file_path, block_size, mode)
    dataset = PiTinyDataset(tokenizer, file_path)
    # dataset = UniversalDataset(tokenizer, file_path="./data_download/memory/TheLastCipher.txt", block_size=256, overlap=64)
    return dataset
