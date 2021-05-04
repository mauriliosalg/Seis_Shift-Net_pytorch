#-*-coding:utf-8-*-
import torch.utils.data
from data.base_data_loader import BaseDataLoader

def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode == 'aligned':
        from data.aligned_dataset import AlignedDataset
        dataset = AlignedDataset()

    elif opt.dataset_mode == 'aligned_resized':
        from data.aligned_dataset_resized import AlignedDatasetResized
        dataset = AlignedDatasetResized()

    elif opt.dataset_mode == 'single':
        from data.single_dataset import SingleDataset
        dataset = SingleDataset()
    
    elif opt.dataset_mode == 'seismic':
        from data.seismic_dataset import SeismicDataset
        dataset = SeismicDataset()

    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        #CH: Inclusão de dataset de validação
        if self.opt.val:
            self.train_set, self.val_set = torch.utils.data.random_split(self.dataset, [int(len(self.dataset)*(1-opt.valsize)), int(len(self.dataset)*opt.valsize)])
            
            self.valdataloader = torch.utils.data.DataLoader(
                self.val_set,
                batch_size=opt.batchSize,
                shuffle=not opt.serial_batches,
                num_workers=int(opt.nThreads))

            self.traindataloader = torch.utils.data.DataLoader(
                self.train_set,
                batch_size=opt.batchSize,
                shuffle=not opt.serial_batches,
                num_workers=int(opt.nThreads))

        else:
            self.traindataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batchSize,
                shuffle=not opt.serial_batches,
                num_workers=int(opt.nThreads))

    def load_data(self):
        if self.opt.val:
            return self.traindataloader , self.valdataloader
        else:    
            return self.traindataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
    
    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i*self.opt.batchSize >= self.opt.max_dataset_size:
                break
            yield data