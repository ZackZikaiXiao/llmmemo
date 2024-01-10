"""TODO(race): Add a description here."""


import json
import random

import datasets


_CITATION = """\
@article{lai2017large,
    title={RACE: Large-scale ReAding Comprehension Dataset From Examinations},
    author={Lai, Guokun and Xie, Qizhe and Liu, Hanxiao and Yang, Yiming and Hovy, Eduard},
    journal={arXiv preprint arXiv:1704.04683},
    year={2017}
}
"""

_DESCRIPTION = """\
Race is a large-scale reading comprehension dataset with more than 28,000 passages and nearly 100,000 questions. The
 dataset is collected from English examinations in China, which are designed for middle school and high school students.
The dataset can be served as the training and test sets for machine comprehension.

"""

_URL = "http://www.cs.cmu.edu/~glai1/data/race/RACE.tar.gz"


class Race(datasets.GeneratorBasedBuilder):
    """ReAding Comprehension Dataset From Examination dataset from CMU"""

    VERSION = datasets.Version("0.1.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="high", description="Exams designed for high school students", version=VERSION),
        datasets.BuilderConfig(
            name="middle", description="Exams designed for middle school students", version=VERSION
        ),
        datasets.BuilderConfig(
            name="all", description="Exams designed for both high school and middle school students", version=VERSION
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                {
                    "example_id": datasets.Value("string"),
                    "article": datasets.Value("string"),
                    "answer": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "options": datasets.features.Sequence(datasets.Value("string"))
                    # These are the features of your dataset like images, labels ...
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="http://www.cs.cmu.edu/~glai1/data/race/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # Downloads the data and defines the splits
        # dl_manager is a datasets.download.DownloadManager that can be used to
        archive = dl_manager.download(_URL)
        case = str(self.config.name)
        if case == "all":
            case = ""
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"train_test_or_eval": f"RACE/test/{case}", "files": dl_manager.iter_archive(archive)},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"train_test_or_eval": f"RACE/train/{case}", "files": dl_manager.iter_archive(archive)},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"train_test_or_eval": f"RACE/dev/{case}", "files": dl_manager.iter_archive(archive)},
            ),
        ]

    def _generate_examples(self, train_test_or_eval, files):
        """Yields examples."""
        for file_idx, (path, f) in enumerate(files):
            if path.startswith(train_test_or_eval) and path.endswith(".txt"):
                data = json.loads(f.read().decode("utf-8"))
                questions = data["questions"]
                answers = data["answers"]
                options = data["options"]
                for i in range(len(questions)):
                    question = questions[i]
                    answer = answers[i]
                    option = options[i]
                    yield f"{file_idx}_{i}", {
                        "example_id": data["id"],
                        "article": data["article"],
                        "question": question,
                        "answer": answer,
                        "options": option,
                    }

def generate_original_data_as_json():
    dataset = datasets.load_dataset('race', 'middle')
    train_data = []
    test_data = []
    dev_data = []

    for dataitem in dataset['train']:
        instance = {}
        instance['article'] = dataitem['article']
        instance['answer'] = dataitem['answer']
        instance['question'] = dataitem['question']
        instance['options'] = dataitem['options']
        train_data.append(instance)
    for dataitem in dataset['test']:
        instance = {}
        instance['article'] = dataitem['article']
        instance['answer'] = dataitem['answer']
        instance['question'] = dataitem['question']
        instance['options'] = dataitem['options']
        test_data.append(instance)
    for dataitem in dataset['validation']:
        instance = {}
        instance['article'] = dataitem['article']
        instance['answer'] = dataitem['answer']
        instance['question'] = dataitem['question']
        instance['options'] = dataitem['options']
        dev_data.append(instance)    
    with open("./data_download/race/original_train.json", 'w') as write_f:
        write_f.write(json.dumps(train_data, indent=4))
    with open("./data_download/race/original_dev.json", 'w') as write_f:
        write_f.write(json.dumps(dev_data, indent=4))
    with open("./data_download/race/original_test.json", 'w') as write_f:
        write_f.write(json.dumps(test_data, indent=4))
    
def generate_train_from_original_train_json(change_train_ratio):
    data_file = open("./data_download/race/original_train.json", 'r')
    content = data_file.read()
    original_data = json.loads(content)
    # total_num_samples = len(original_data)
    if change_train_ratio:
        num_for_each_label = [11555, 8088, 4622, 1156]
    else:
        num_for_each_label = [6356, 6355, 6355, 6355]
    label_to_be_filled = 'A'
    alphabet=['A', 'B', 'C', 'D']
    correct_answer_id_num = {
        'A':0,
        'B':0,
        'C':0,
        'D':0,
    }
    random.shuffle(original_data)
    generated_data = []
    for item in original_data:
        if not (item['answer'] == label_to_be_filled):
            item['options'][ord(label_to_be_filled) - ord('A')], item['options'][ord(item['answer']) - ord('A')] = item['options'][ord(item['answer']) - ord('A')], item['options'][ord(label_to_be_filled) - ord('A')]
            item['answer'] = label_to_be_filled
        num_for_each_label[ord(label_to_be_filled) - ord('A')] -= 1
        if num_for_each_label[ord(label_to_be_filled) - ord('A')] == 0 and ord(label_to_be_filled) <= ord('D'):
                label_to_be_filled = chr(ord(label_to_be_filled) + 1)
        instance = {}
        instance['instruction'] = item['question']
        instance['context'] = item['article']
        for index, content in enumerate(item['options']):
            if index == 0:
                instance['instruction'] = instance['instruction'] + " options: "+ alphabet[index] + ':' + content
            else:
                instance['instruction'] += " " + alphabet[index] + ':' + content
        instance['response'] = item['answer']
        correct_answer_id_num[item['answer']] += 1
        instance['category'] = 'race'
        generated_data.append(instance)
    data_file.close()
    random.shuffle(generated_data)
    print(correct_answer_id_num)
    with open("./data_download/race/train.json", 'w') as write_f:
        write_f.write(json.dumps(generated_data, indent=4))
            

if __name__ == "__main__":
    # generate_original_data_as_json()
    generate_train_from_original_train_json(False)
    # print(chr(ord('A')+1))