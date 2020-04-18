import re
import numpy as np

class tokenizer():
    def __init__(self, data):
        self.data_fixed = []
        self.pad_string = 'PAD'
        #fix Br Cl issue
        for d in data:
            s = d[0]
            target = d[1]
            s = self.fixBrCl(s)
            self.data_fixed.append([s, target])
        self.customAtoms = set()
        for d in self.data_fixed:
            s = d[0]
            ca = re.findall('\[(.*?)\]', s)
            for c in ca:
                self.customAtoms.add(c)
        self.split_data = []
        for d in self.data_fixed:
            self.split_data.append(self.splitString(d[0]))

        self.token_set = set()
        self.max_length = 0
        for d in self.split_data:
            length = len(d)
            if self.max_length < length:
                self.max_length = length
            for x in d:
                self.token_set.add(x)
        self.token_set.add(self.pad_string)
        print('Number of tokens:', len(self.token_set))
        print('Max length:', self.max_length)
        
        self.id2token = dict(enumerate(self.token_set))
        self.token2id = {token: i for i, token in self.id2token.items()}
        
    def fixBrCl(self, s):
        element_map = {'Cl': 'K', 'Br': 'R'}
        for key, value in element_map.items():
            s = s.replace(key, value)
        return s

    def splitCharge(self, line, d):
        s = line.split(d)
        out = []
        if len(s) > 1:
            for i, k in enumerate(s):
                if i > 0:
                    out.append(d + k)
                else:
                    out.append(k)
        return out
    
    def splitString(self, input):
        splits = re.split('\[(.*?)\]', input)
        final = []
        for s in splits:
            if s in self.customAtoms:
                if '-' in s:
                    final += self.splitCharge(s, '-')
                elif '+' in s:
                    final += self.splitCharge(s, '+')
                else:
                    final.append(s)
                # final.append(s)
            else:
                for c in s:
                    final.append(c)
        return final

    def tokenize(self, s):
        s = self.fixBrCl(s)
        s = self.splitString(s)
        tokenized = [self.token2id[x] for x in s]
        len_tokenized = len(tokenized)
        for _ in range(0, self.max_length - len_tokenized):
            tokenized.append(self.token2id[self.pad_string])
        return np.array(tokenized, dtype=np.long), len_tokenized

    def vocab_len(self):
        return len(self.token_set)
