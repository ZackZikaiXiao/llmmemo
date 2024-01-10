import os
from data_download.databricks_dolly_15k.client_data_allocation import allocate_data
from data_download.GLUE.partition import partition


class DataPartition:
    def __init__(self, args) -> None:
        self.dataset = args.dataset
        self.num_client = args.num_clients
        self.dirichlet_alpha = args.dirichlet_alpha
        self.partition_method = args.partition_method
        self.data_path = args.data_path

    def partition(self):
        data_folders = {
            "quail": "./data_download/quail/train.json",
            "new-databricks-dolly-15k": './data_download/databricks_dolly_15k/data',
            "sst-2": "./data_download/GLUE/sst-2/SST-2/SST-2.json",
            "rte": "./data_download/GLUE/rte/RTE/RTE.json",
            "cola": "./data_download/GLUE/cola/CoLA/CoLA.json",
            "qnli": "./data_download/GLUE/qnli/QNLI/QNLI.json",
            "qqp": "./data_download/GLUE/qqp/QQP/QQP.json",
            "sts-b": "./data_download/GLUE/sts-b/STS-B/STS-B.json",
            "wnli": "./data_download/GLUE/wnli/WNLI/WNLI.json",
            "mrpc": "./data_download/GLUE/mrpc/MRPC/MRPC.json",
            "mnli": "./data_download/GLUE/mnli/MNLI/MNLI.json",
        }
        GLUE_dataset =["sst-2", "rte", "cola", "qnli", "qqp", "sts-b", "wnli", "mrpc", "mnli"]
        data_folder = data_folders[self.dataset]
        if self.dataset == "new-databricks-dolly-15k":
            
            # data_folder = './data_download/databricks_dolly_15k/data'
            if os.path.exists(data_folder):
                print(f"new-databricks-dolly-15k is ready.")
            else:
                diff_quantity = 0  # Whether clients have different amounts of data
                # 直接调用函数
                allocate_data(self.num_client, diff_quantity)
        elif self.dataset in GLUE_dataset:
            # data_folder = data_folders[self.dataset]
            partition(data_path=data_folder, save_path=self.data_path,num_clients=self.num_client, dirichlet_alpha=self.dirichlet_alpha,
                      partition_method=self.partition_method, num_of_classes_for_stsb=3)
        elif self.dataset == "quail":
            partition(data_path=data_folder, save_path=self.data_path,num_clients=self.num_client, dirichlet_alpha=self.dirichlet_alpha,
                      partition_method=self.partition_method)
            
def partition_data(args):
    data_partition = DataPartition(args)
    data_partition.partition()