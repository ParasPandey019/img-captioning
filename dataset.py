import json
import os
import nltk
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, BatchSampler
from tqdm import tqdm
from vocab import Vocabulary


class CoCoDataset(Dataset):
    """
    COCO Dataset for image captioning.
    """
    def __init__(self, transform, mode='train', batch_size=32, 
                 vocab_threshold=5, vocab_file='./vocab.pkl',
                 start_word="<start>", end_word="<end>", unk_word="<unk>",
                 vocab_from_file=True, cocoapi_loc='./coco2017/'):
        """
        Initialize COCO dataset.
        
        Args:
            transform: Image transformations
            mode (str): 'train', 'valid', or 'test'
            batch_size (int): Batch size for training
            vocab_threshold (int): Minimum word count threshold
            vocab_file (str): Vocabulary file path
            start_word (str): Start token
            end_word (str): End token
            unk_word (str): Unknown token
            vocab_from_file (bool): Load vocabulary from file
            cocoapi_loc (str): Path to COCO dataset
        """
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        
        # Set paths based on mode
        if mode == 'train':
            self.img_folder = os.path.join(cocoapi_loc, 'train2017/')
            annotations_file = os.path.join(cocoapi_loc, 'annotations/captions_train2017.json')
        elif mode == 'valid':
            self.img_folder = os.path.join(cocoapi_loc, 'val2017/')
            annotations_file = os.path.join(cocoapi_loc, 'annotations/captions_val2017.json')
        elif mode == 'test':
            self.img_folder = os.path.join(cocoapi_loc, 'test2017/')
            annotations_file = os.path.join(cocoapi_loc, 'annotations/image_info_test2017.json')
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        # Create vocabulary
        self.vocab = Vocabulary(vocab_threshold, vocab_file, start_word, end_word, 
                               unk_word, os.path.join(cocoapi_loc, 'annotations/captions_train2017.json'), 
                               vocab_from_file)
        
        # Load dataset based on mode
        if mode == 'train':
            self.coco = COCO(annotations_file)
            self.ids = list(self.coco.anns.keys())
            print("Obtaining caption lengths...")
            tokenized_captions = [
                nltk.tokenize.word_tokenize(str(self.coco.anns[self.ids[index]]['caption']).lower())
                for index in tqdm(range(len(self.ids)))
            ]
            self.caption_lengths = [len(tokens) for tokens in tokenized_captions]
        else:
            with open(annotations_file, 'r') as f:
                test_info = json.load(f)
            self.paths = [item['file_name'] for item in test_info['images']]
    
    def __getitem__(self, index):
        """Get dataset item."""
        if self.mode == 'train':
            ann_id = self.ids[index]
            caption = self.coco.anns[ann_id]['caption']
            img_id = self.coco.anns[ann_id]['image_id']
            path = self.coco.loadImgs(img_id)[0]['file_name']
            
            # Load and transform image
            image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
            image = self.transform(image)
            
            # Convert caption to tensor
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            caption_tokens = [self.vocab(self.vocab.start_word)]
            caption_tokens.extend([self.vocab(token) for token in tokens])
            caption_tokens.append(self.vocab(self.vocab.end_word))
            caption = torch.tensor(caption_tokens, dtype=torch.long)
            
            return image, caption
            
        elif self.mode == 'valid':
            path = self.paths[index]
            image_id = int(path.split('.')[0].split('_')[-1])
            
            # Load and transform image
            image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
            image = self.transform(image)
            
            return image_id, image
            
        else:  # test mode
            path = self.paths[index]
            
            # Load and transform image
            pil_image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
            orig_image = np.array(pil_image)
            image = self.transform(pil_image)
            
            return orig_image, image
    
    def get_train_indices(self):
        """Get training indices with same caption length."""
        sel_length = np.random.choice(self.caption_lengths)
        all_indices = np.where([
            self.caption_lengths[i] == sel_length 
            for i in range(len(self.caption_lengths))
        ])[0]
        indices = list(np.random.choice(all_indices, size=self.batch_size))
        return indices
    
    def __len__(self):
        """Return dataset length."""
        if self.mode == 'train':
            return len(self.ids)
        else:
            return len(self.paths)


def get_loader(transform, mode='train', batch_size=32, vocab_threshold=5,
               vocab_file='./vocab.pkl', start_word="<start>", end_word="<end>",
               unk_word="<unk>", vocab_from_file=True, num_workers=0, 
               cocoapi_loc='./coco2017/'):
    """
    Create data loader for COCO dataset.
    
    Args:
        transform: Image transformations
        mode (str): 'train', 'valid', or 'test'
        batch_size (int): Batch size
        vocab_threshold (int): Minimum word count threshold
        vocab_file (str): Vocabulary file path
        start_word (str): Start token
        end_word (str): End token  
        unk_word (str): Unknown token
        vocab_from_file (bool): Load vocabulary from file
        num_workers (int): Number of data loading workers
        cocoapi_loc (str): Path to COCO dataset
        
    Returns:
        DataLoader: PyTorch data loader
    """
    dataset = CoCoDataset(transform=transform, mode=mode, batch_size=batch_size,
                         vocab_threshold=vocab_threshold, vocab_file=vocab_file,
                         start_word=start_word, end_word=end_word, unk_word=unk_word,
                         vocab_from_file=vocab_from_file, cocoapi_loc=cocoapi_loc)
    
    if mode == 'train':
        indices = dataset.get_train_indices()
        initial_sampler = SubsetRandomSampler(indices=indices)
        data_loader = DataLoader(dataset=dataset, num_workers=num_workers,
                                batch_sampler=BatchSampler(sampler=initial_sampler,
                                                         batch_size=dataset.batch_size,
                                                         drop_last=False))
    else:
        data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True,
                                num_workers=num_workers)
    
    return data_loader
