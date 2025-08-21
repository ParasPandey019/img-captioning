import os
import pickle
from collections import Counter
import nltk
from pycocotools.coco import COCO


class Vocabulary:
    """
    Vocabulary class for managing word-to-index mappings.
    """
    def __init__(self, vocab_threshold=5, vocab_file="./vocab.pkl", 
                 start_word="<start>", end_word="<end>", unk_word="<unk>",
                 annotations_file="./coco2017/annotations/captions_train2017.json", 
                 vocab_from_file=True):
        """
        Initialize the vocabulary.
        
        Args:
            vocab_threshold (int): Minimum word count threshold
            vocab_file (str): Path to save/load vocabulary
            start_word (str): Start-of-sentence token
            end_word (str): End-of-sentence token  
            unk_word (str): Unknown word token
            annotations_file (str): Path to COCO annotations file
            vocab_from_file (bool): Whether to load from existing file
        """
        self.vocab_threshold = vocab_threshold
        self.vocab_file = vocab_file
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.annotations_file = annotations_file
        self.vocab_from_file = vocab_from_file
        
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        
        self.get_vocab()
    
    def get_vocab(self):
        """Load vocabulary from file or build from scratch."""
        if os.path.exists(self.vocab_file) and self.vocab_from_file:
            with open(self.vocab_file, 'rb') as f:
                vocab = pickle.load(f)
                self.word2idx = vocab.word2idx
                self.idx2word = vocab.idx2word
                print("Vocabulary successfully loaded from vocab.pkl file!")
        else:
            self.build_vocab()
            with open(self.vocab_file, 'wb') as f:
                pickle.dump(self, f)
                print("Vocabulary saved to vocab.pkl file!")
    
    def build_vocab(self):
        """Build vocabulary from COCO captions."""
        print("Building vocabulary from scratch...")
        self.init_vocab()
        self.add_word(self.start_word)
        self.add_word(self.end_word)
        self.add_word(self.unk_word)
        self.add_captions()
    
    def init_vocab(self):
        """Initialize vocabulary dictionaries."""
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
    
    def add_word(self, word):
        """Add a word to vocabulary."""
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def add_captions(self):
        """Process COCO captions and add words to vocabulary."""
        coco = COCO(self.annotations_file)
        counter = Counter()
        ids = coco.anns.keys()
        
        for i, idx in enumerate(ids):
            caption = str(coco.anns[idx]['caption'])
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)
            
            if i % 100000 == 0:
                print(f'[{i}/{len(ids)}] Tokenizing captions...')
        
        # Keep only words that meet threshold
        words = [word for word, cnt in counter.items() if cnt >= self.vocab_threshold]
        
        for word in words:
            self.add_word(word)
        
        print(f'Total vocabulary size: {len(self.word2idx)}')
    
    def __call__(self, word):
        """Get word index, return <unk> index if word not found."""
        if word not in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]
    
    def __len__(self):
        """Return vocabulary size."""
        return len(self.word2idx)
