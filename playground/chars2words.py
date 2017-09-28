#%%
#cd ../WikiParsing
#%%
#from wiki import extraction
#wikidf = extraction.extract_wiki_dump_text_only("../data/dawiki-20170820-pages-meta-current.xml")
#%%
#wikidf.head()
#%%
with open("/home/jonatan/Dropbox/europarl/europarl-v7.da-en.da", 'r') as f:
    da = f.readlines()
with open("/home/jonatan/Dropbox/europarl/europarl-v7.da-en.en", 'r') as f:
    en = f.readlines()

#%%
import pandas as pd
europarldf = pd.DataFrame(dict(da=da, en=en))
print(europarldf.shape)
europarldf.head()
#%%

#%%
cd ../Mowgly
#%%
from mowgly import text
#%%
# column_specs = [
#    text.CharColumnSpec(column_name='text', language='danish', lower=True,
#                        min_len=200, max_len=2000,
#                        fixed_length_subsample=True, uniform_start=True, sample_len=100),
#    text.CharColumnSpec(column_name='orig_title', language='danish', lower=True,
#                        min_len=1, max_len=100)]
#wiki_colstore = text.TextColumnStore(corpus=wikidf, column_specs=column_specs, n_jobs=8, copy=False)
#%%
column_specs = [
    text.CharColumnSpec(column_name='da', language='danish', lower=True,
                        min_len=1, max_len=200),
    text.CharColumnSpec(column_name='en', language='english', lower=True,
                        min_len=1, max_len=200)]
europarl_colstore = text.TextColumnStore(corpus=europarldf, column_specs=column_specs, n_jobs=8, copy=False)
#%%
from mowgly.training.generators import DefaultGeneratorFactory, take_sub_arrays
from mowgly import training
from mowgly.model import Model
from mowgly.text.utils import get_padded_whitespace_indices
import pandas as pd
import numpy as np
#%%
#class MyGeneratorFactory(DefaultGeneratorFactory):
#
#    def make_train_generator(self):
#
#        random_indices_0 = self.random_state.choice(self.train_indices, size=self.train_size, replace=True)
#        random_indices_1 = self.random_state.choice(self.train_indices, size=self.train_size, replace=True)
#
###            sub_indices_1 = random_indices_1[i:i + self.train_minibatch_size]
            # yield {'sample_0': take_sub_arrays(self.arrays, sub_indices_0),
            #        'sample_1': take_sub_arrays(self.arrays, sub_indices_1)
            #        }
#%%

train_indices, val_indices = training.utils.get_train_test_random_indices(
                        num_samples=europarl_colstore.size, test_ratio=0.1)

#%%
#al_counts = colstore.corpus.title.value_counts().astype(np.float64)
#sample_probs = val_counts[colstore.corpus.title.values].values[train_indices]
#sample_probs /= sample_probs.sum()

from torch import nn
#%%
cd ../fairytech/thlib
#%%
from thlib.functions import select_at
from thlib.layers.n_step_lstm import NStepLSTM
from thlib.layers.n_step_linear import NStepLinear
#%%
class CharsEncoder(nn.Module):

    def __init__(self, vocabsize, embedsize, outsize):

        super(CharsEncoder, self).__init__()

        assert outsize%4 == 0 and outsize > 0, "outsize must be divisible by 4"
        self.embeds = nn.Embedding(num_embeddings=vocabsize, embedding_dim=embedsize)

        self.convs = nn.ModuleList([
                    nn.Conv2d(in_channels=1, out_channels=outsize//4, kernel_size=(3,embedsize), padding=(1,0)),
                    nn.Conv2d(in_channels=1, out_channels=outsize//4, kernel_size=(5,embedsize), padding=(2,0)),
                    nn.Conv2d(in_channels=1, out_channels=outsize//4, kernel_size=(7,embedsize), padding=(3,0)),
                    nn.Conv2d(in_channels=1, out_channels=outsize//4, kernel_size=(9,embedsize), padding=(4,0)),
                    ])

    def forward(self, text_ids):
        embed = self.embeds(text_ids)[:,None]
        out = torch.cat(
            [conv(embed) for conv in self.convs], dim=1
        ).permute(3,0,2,1)

        return out[0].contiguous()
#%%

class WordsEncoder(nn.Module):

    def __init__(self, insize, outsize):

        super(WordsEncoder, self).__init__()

        self.linear = NStepLinear(in_features=insize*3, out_features=outsize, bias=False)

    def forward(self, chars, startword_indices, midword_indices, endword_indices, num_words):

        chars_start_att = select_at(x=chars, padded_indices=startword_indices, num_indices=num_words, fill_value=0)
        chars_mid_att = select_at(x=chars, padded_indices=midword_indices, num_indices=num_words, fill_value=0)
        chars_end_att = select_at(x=chars, padded_indices=endword_indices, num_indices=num_words, fill_value=0)

        words = self.linear(
                torch.cat([chars_start_att, chars_mid_att, chars_end_att], dim=2),
                )

        return words

#%%
class Chars2WordsEncoder(nn.Module):

    def __init__(self, vocabsize, char_embed_size=16, chars_encode_size=256, word_encode_size=256):

        super(Chars2WordsEncoder, self).__init__()

        self.chars_encoder = CharsEncoder(vocabsize=vocabsize,
                                         embedsize=char_embed_size, outsize=chars_encode_size)

        self.words_encoder = WordsEncoder(insize=chars_encode_size, outsize=word_encode_size)

    def forward(self, text_ids, startword_indices, midword_indices, end_indices, num_words):
        chars = self.chars_encoder(text_ids)
        words = self.words_encoder(chars, startword_indices, midword_indices, end_indices, num_words)
        return words
#%%


class Flipper(object):

    def __init__(self, dim):
        self.dim = dim
        self._reverse_indices = None

    def __call__(self, tensor):
        L = tensor.size(self.dim)
        if self._reverse_indices is None or self._arange_length < L:
            self._reverse_indices = torch.arange(L - 1, -1, -1).long()
            self._arange_length = L
        if tensor.is_cuda and not self._reverse_indices.is_cuda:
            self._reverse_indices = self._reverse_indices.cuda()
        elif  self._reverse_indices.is_cuda and not tensor.is_cuda:
            self._reverse_indices = self._reverse_indices.cpu()
        reversed_tensor = tensor.index_select(self.dim, Variable(self._reverse_indices[max(self._arange_length-L,0):]))
        return reversed_tensor

class SentenceEncoder(nn.Module):

    def __init__(self, insize, outsize):

        super(SentenceEncoder, self).__init__()

        self.fwd_lstm = NStepLSTM(in_features=insize, out_features=outsize)
        self.bwd_lstm = NStepLSTM(in_features=insize, out_features=outsize)
        self.outsize = outsize
        self.flipper = Flipper(dim=0)

    def _run_lstm(self, words, direction):
        if direction == 'forward':
            layer = self.fwd_lstm
        elif direction == 'backward':
            layer = self.bwd_lstm
            words = self.flipper(words)
        else:
            raise ValueError('unsupported direction')
        h0 = Variable(words.data.new(words.size(1),self.outsize).zero_())
        c0 = Variable(words.data.new(words.size(1),self.outsize).zero_())
        return layer(words, h0, c0)

    def forward(self, words):
        fwd_h, c0 = self._run_lstm(words, 'forward')
        bwd_h, c0 = self._run_lstm(words, 'backward')
        return (fwd_h + bwd_h).sum(dim=0)



#%%

class TwoLangChars2Words2SentenceEncoder(nn.Module):

    def __init__(self, lang0_vocabsize, lang1_vocabsize, char_embed_size=16, chars_encode_size=256,
                 word_encode_size=256, sent_encode_size=512):

        super(TwoLangChars2Words2SentenceEncoder, self).__init__()

        self.lang0_encoder = Chars2WordsEncoder(vocabsize=lang0_vocabsize,
                                                char_embed_size=char_embed_size, chars_encode_size=chars_encode_size,
                                                word_encode_size=word_encode_size)

        self.lang1_encoder = Chars2WordsEncoder(vocabsize=lang1_vocabsize,
                                                char_embed_size=char_embed_size, chars_encode_size=chars_encode_size,
                                                word_encode_size=word_encode_size)

        self.sent_encoder = SentenceEncoder(insize=word_encode_size, outsize=sent_encode_size)

    def forward(self, lang0_args, lang1_args):
        words_lang0 = self.lang0_encoder(*lang0_args)
        num_words_lang0 = words_lang0.size(1)
        words_lang1 = self.lang1_encoder(*lang1_args)
        words = torch.cat([words_lang0, words_lang1], dim=1).permute(1,0,2).contiguous()
        sentence = self.sent_encoder(words).tanh()
        sentence_lang0, sentence_lang0 = sentence[:num_words_lang0], sentence[num_words_lang0:]
        return sentence_lang0, sentence_lang0

#%%
import torch
from torch.autograd import Variable
#%%
def get_encoder_inputs(text_id_vectors, whitespace_ids, eos_id, on_gpu=False):
    whitespace_indices, num_whitespaces = get_padded_whitespace_indices(text_id_vectors, whitespace_ids, eos_id)
    startword_indices = whitespace_indices[:,:-1] + 1
    half_length = (whitespace_indices[:,1:] - whitespace_indices[:,:-1]) // 2
    midword_indices = whitespace_indices[:,:-1] + half_length
    endword_indices = midword_indices + half_length
    num_words = num_whitespaces - 1
    text_ids = torch.from_numpy(text_id_vectors)
    startword_indices = torch.from_numpy(startword_indices)
    midword_indices = torch.from_numpy(midword_indices)
    endword_indices = torch.from_numpy(endword_indices)
    num_words = torch.from_numpy(num_words)
    if on_gpu:
        text_ids = text_ids.cuda()
        startword_indices = startword_indices.cuda()
        midword_indices = midword_indices.cuda()
        num_words = num_words.cuda()
        endword_indices = endword_indices.cuda()

    text_ids = Variable(text_ids.long())
    startword_indices = Variable(startword_indices)
    midword_indices = Variable(midword_indices)
    endword_indices = Variable(endword_indices)
    num_words = Variable(num_words)

    return text_ids, startword_indices, midword_indices, endword_indices, num_words

#%%

twolangchars2words2sentence = TwoLangChars2Words2SentenceEncoder(
                                                   lang0_vocabsize=europarl_colstore.specs['da'].vocabsize,
                                                   lang1_vocabsize=europarl_colstore.specs['en'].vocabsize,
                                                   char_embed_size=8, chars_encode_size=256,
                                                   word_encode_size=256, sent_encode_size=512)


from torch.optim import Adam

optimizer = Adam(params=twolangchars2words2sentence.parameters())
#%%

genfactory = DefaultGeneratorFactory(
                    arrays=dict(europarl_colstore=europarl_colstore),
                    train_indices=train_indices.astype(np.int32), val_indices=val_indices.astype(np.int32),
                    train_minibatch_size=192)
on_gpu = True

if on_gpu:
    twolangchars2words2sentence.cuda()
else:
    twolangchars2words2sentence.cpu()
from torch.nn.functional import cosine_embedding_loss
#%%
for i in range(100):
    total_loss = 0
    gen = genfactory.make_train_generator()
    for k, arrays in enumerate(gen):

        da_input = get_encoder_inputs(text_id_vectors=arrays['europarl_colstore']['da']['id_vectors'],
                                      whitespace_ids=europarl_colstore.specs['da'].whitespace_ids,
                                      eos_id=europarl_colstore.specs['da'].eos_id,
                                      on_gpu=on_gpu)
        en_input = get_encoder_inputs(text_id_vectors=arrays['europarl_colstore']['en']['id_vectors'],
                                      whitespace_ids=europarl_colstore.specs['da'].whitespace_ids,
                                      eos_id=europarl_colstore.specs['en'].eos_id,
                                      on_gpu=on_gpu)
        sentence_lang0, sentence_lang1 = twolangchars2words2sentence(da_input, en_input)


        if on_gpu:
             similar_target = Variable(torch.cuda.LongTensor(len(sentence_lang0)).fill_(1))
        else:
             similar_target = Variable(torch.LongTensor(len(sentence_lang0)).fill_(1))

        loss = cosine_embedding_loss(sentence_lang0, sentence_lang1, similar_target)


        # sentence_0 = get_sentence_encoding(encoder=chars2words2sentence,
        #                       text_id_vectors=arrays['sample_0']['colstore']['text']['id_vectors'],
        #                       whitespace_ids=colstore.specs['text'].whitespace_ids,
        #                       eos_id=colstore.specs['text'].eos_id,
        #                       on_gpu=on_gpu)
        #
        # sentence_1 = get_sentence_encoding(encoder=chars2words2sentence,
        #                       text_id_vectors=arrays['sample_1']['colstore']['text']['id_vectors'],
        #                       whitespace_ids=colstore.specs['text'].whitespace_ids,
        #                       eos_id=colstore.specs['text'].eos_id,
        #                       on_gpu=on_gpu)
        #
        # title_0 = get_sentence_encoding(encoder=chars2words2sentence,
        #                       text_id_vectors=arrays['sample_0']['colstore']['orig_title']['id_vectors'],
        #                       whitespace_ids=colstore.specs['orig_title'].whitespace_ids,
        #                       eos_id=colstore.specs['orig_title'].eos_id,
        #                       on_gpu=on_gpu)
        #
        # title_1 = get_sentence_encoding(encoder=chars2words2sentence,
        #                       text_id_vectors=arrays['sample_1']['colstore']['orig_title']['id_vectors'],
        #                       whitespace_ids=colstore.specs['orig_title'].whitespace_ids,
        #                       eos_id=colstore.specs['orig_title'].eos_id,
        #                       on_gpu=on_gpu)
        # if on_gpu:
        #     similar_target = Variable(torch.cuda.LongTensor(len(sentence_0)).fill_(1))
        #     dissimilar_target = Variable(torch.cuda.LongTensor(len(sentence_0)).fill_(-1))
        # else:
        #     similar_target = Variable(torch.LongTensor(len(sentence_0)).fill_(1))
        #     dissimilar_target = Variable(torch.LongTensor(len(sentence_0)).fill_(-1))
        #
        # loss = ( cosine_embedding_loss(sentence_0, title_0, similar_target)
        #        + cosine_embedding_loss(sentence_1, title_1, similar_target)
        #        + cosine_embedding_loss(sentence_1, title_0, dissimilar_target)
        #        + cosine_embedding_loss(sentence_0, title_1, dissimilar_target)
        #        )
        total_loss += loss.data
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("loss at iter {0}: {1}".format(i, total_loss.cpu().numpy()[0]))

#%%
total_loss
