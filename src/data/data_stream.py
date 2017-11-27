import logging

from fuel import config
from fuel.transformers import (SourcewiseTransformer,
                               Transformer,
                               AgnosticTransformer,
                               FilterSources,
                               Rename,
                               Padding)
from fuel.schemes import (SequentialScheme,
                          ShuffledScheme)
from fuel.streams import DataStream
import numpy as np

logger = logging.getLogger(__name__)


def liacl_data_stream(dataset, batch_size, vocab, rel_vocab, target='negative_sampling',
                      name=None, k=3, shuffle=False, neg_sample_kwargs={}):
    batches_per_epoch = int(np.ceil(dataset.num_examples / float(batch_size)))
    if shuffle:
        iteration_scheme = ShuffledScheme(dataset.num_examples, batch_size)
    else:
        iteration_scheme = SequentialScheme(dataset.num_examples, batch_size)

    data_stream = DataStream(dataset, iteration_scheme=iteration_scheme)
    data_stream = NumberizeWords(data_stream, vocab, which_sources=('head', 'tail'))
    data_stream = NumberizeWords(data_stream, rel_vocab, which_sources=('rel'))

    if target == "score":
        data_stream = Rename(data_stream, {'score': 'target'})
    else:
        data_stream = FilterSources(data_stream, sources=('head', 'tail', 'rel'))

    data_stream = Padding(data_stream, mask_sources=('head, tail'), mask_dtype=np.float32)

    if target == 'negative_sampling':
        logger.info('target for data stream ' + name + ' is negative sampling')
        data_stream = NegativeSampling(data_stream, k=k)
    elif target == 'filtered_negative_sampling':
        logger.info('target for data stream ' + name + ' is filtered negative sampling')
        data_stream = FilteredNegativeSampling(data_stream, k=k, **neg_sample_kwargs)
    elif target == 'score':
        logger.info('target for data stream ' + str(name) + ' is score')
    else:
        raise NotImplementedError('target ', target, ' must be one of "score" or "negative_sampling"')

    data_stream = MergeSource(data_stream, merge_sources=('head', 'tail', 'head_mask', 'tail_mask', 'rel'),
                              merge_name='input')

    return data_stream, batches_per_epoch


def _liacl_add_closest_neighbour(stream):
    """
    Adds closest neighbour to stream by first collecting 1 epoch of stream, ignoring negative samples
    """
    raise NotImplementedError()



class NumberizeWords(SourcewiseTransformer):
    def __init__(self, data_stream, vocab):
        super(NumberizeWords, self).__init__(data_stream,
                                             produces_examples=data_stream.produces_examples,
                                             *args,
                                             **kwargs)
        self.vocab = vocab

    def transform_source_batch(self, source_batch, source_name):
        return np.array([self.vocab.encode(words) for words in source_batch])


class FilteredNegativeSampling(Transformer):
    """
    Params
    ------
    filter_fnc: function
    Function taking in head, rel, tail matrix and producing 1 and 0 vector indicating if we
    accept this sample or not
    """

    def __init__(self, data_stream, filter_fnc, k=3, *args, **kwargs):
        if data_stream.produces_examples:
            raise ValueError('the wrapped data stream must produce batches, '
                             'not examples.')

        self.rng = kwargs.pop('rng', None)
        if self.rng is None:
            self.rng = np.random.RandomState(config.default_seed)

        self.k = k
        self.filter_fnc = filter_fnc

        self.batch_id = 0
        self.samples_per_batch = 0

        super(FilteredNegativeSampling, self).__init__(data_stream,
                                                       produces_examples=False,
                                                       *args,
                                                       **kwargs)

    @property
    def sources(self):
        return self.data_stream.sources + ('target',)

    def transform_batch(self, batch):
        head, head_mask, tail, tail_mask, rel = batch
        batch_size = len(head)

        head_list, head_mask_list, rel_list, tail_list, tail_mask_list, target_list = \
            [head], [head_mask], [rel], [tail], [tail_mask], [np.array([1] * batch_size)]

        k = 0

        # TODO: Parametrize elegantly by resampling procedure, but later?
        # For now just have argsim. Or maybe do not have the resampling?

        # The way it goes is sample in while loop until it found examples fooling ArgSim
        while sum([len(x) for x in head_list]) < (self.k + 1) * batch_size:
            neg_rels_idx_sample = np.random.randint(batch_size, size=batch_size)
            neg_head_idx_sample = np.random.randint(batch_size, size=batch_size)
            neg_tail_idx_sample = np.random.randint(batch_size, size=batch_size)

            neg_rel_sample = rel[neg_rels_idx_sample]
            neg_head_sample = head[neg_head_idx_sample]
            neg_tail_sample = tail[neg_tail_idx_sample]
            neg_head_mask_sample = head_mask[neg_head_idx_sample]
            neg_tail_mask_sample = tail_mask[neg_tail_idx_sample]

            rel_sample = np.concatenate([neg_rel_sample, rel, rel], axis=0).reshape(-1, 1)
            head_sample = np.concatenate([head, neg_head_sample, head], axis=0)
            tail_sample = np.concatenate([tail, tail, neg_tail_sample], axis=0)
            head_mask_sample = np.concatenate([head_mask, neg_head_mask_sample, head_mask], axis=0)
            tail_mask_sample = np.concatenate([tail_mask, tail_mask, neg_tail_mask_sample], axis=0)
            target_sample = np.array([0] * batch_size * 3)

            accept = self.filter_fnc(head_sample, rel_sample, tail_sample)

            assert len(accept) == len(head_sample)

            # .reshape(-1,) to be compatible with array of lists that is sometimes produced
            # if examples have varying length (remember padding is after negsampling ATM)
            # this could be changed easily, no reason to have padding after
            head_list.append(head_sample[accept])
            head_mask_list.append(head_mask_sample[accept])
            rel_list.append(rel_sample[accept])
            tail_list.append(tail_sample[accept])
            tail_mask_list.append(tail_mask_sample[accept])
            target_list.append(target_sample[accept])

            k += 3 * batch_size

        self.samples_per_batch = 0.9 * self.samples_per_batch + 0.1 * k
        if self.batch_id % 100 == 0:
            print("Avg samples_per_batch={}".format(self.samples_per_batch))
            self.batch_id += 1

        rel = np.concatenate(rel_list, axis=0)[0:(self.k + 1) * batch_size]
        head = np.concatenate(head_list, axis=0)[0:(self.k + 1) * batch_size]
        tail = np.concatenate(tail_list, axis=0)[0:(self.k + 1) * batch_size]
        head_mask = np.concatenate(head_mask_list, axis=0)[0:(self.k + 1) * batch_size]
        tail_mask = np.concatenate(tail_mask_list, axis=0)[0:(self.k + 1) * batch_size]
        target = np.concatenate(target_list, axis=0)[0:(self.k + 1) * batch_size]

        return (head, head_mask, tail, tail_mask, rel, target)


class NegativeSampling(Transformer):
    def __init__(self, data_stream, k=3, *args, **kwargs):
        if data_stream.produces_examples:
            raise ValueError('the wrapped data stream must produce batches, '
                             'not examples.')

        self.k = k
        self.rng = kwargs.pop('rng', None)
        if self.rng is None:
            self.rng = np.random.RandomState(config.default_seed)

        super(NegativeSampling, self).__init__(data_stream,
                                               produces_examples=False,
                                               *args,
                                               **kwargs)

    @property
    def sources(self):
        return self.data_stream.sources + ('target',)

    def transform_batch(self, batch):
        head, head_mask, tail, tail_mask, rel = batch
        batch_size = len(head)

        neg_rels_idx = np.random.randint(batch_size, size=batch_size)
        neg_head_idx = np.random.randint(batch_size, size=batch_size)
        neg_tail_idx = np.random.randint(batch_size, size=batch_size)

        neg_rel = rel[neg_rels_idx]
        neg_head = head[neg_head_idx]
        neg_tail = tail[neg_tail_idx]
        neg_head_mask = head_mask[neg_head_idx]
        neg_tail_mask = tail_mask[neg_tail_idx]

        rel = np.concatenate([rel, neg_rel, rel, rel], axis=0)
        head = np.concatenate([head, head, neg_head, head], axis=0)
        tail = np.concatenate([tail, tail, tail, neg_tail], axis=0)
        head_mask = np.concatenate([head_mask, head_mask, neg_head_mask, head_mask], axis=0)
        tail_mask = np.concatenate([tail_mask, tail_mask, tail_mask, neg_tail_mask], axis=0)

        # TODO(kudkudak): This is a terrible hack
        if self.k < 3:
            # Can 1/3 of it be false negative?
            assert len(head) == 4*batch_size
            ids = range(batch_size)
            ids_chosen = np.random.choice(batch_size*3, batch_size*self.k, replace=False)
            ids = ids + [iid + batch_size for iid in ids_chosen]
            rel = rel[ids]
            head = head[ids]
            tail = tail[ids]
            head_mask = head_mask[ids]
            tail_mask = tail_mask[ids]
        elif self.k > 3:
            raise NotImplementedError()

        target = np.array([1] * batch_size + [0] * batch_size * self.k)

        return (head, head_mask, tail, tail_mask, rel, target)


class MergeSource(AgnosticTransformer):
    """ Merge selected sources into a single source

    Merged source becomes {source_name: source,...} for all former sources
    Added to start
    """

    def __init__(self, data_stream, merge_sources, merge_name, *args, **kwargs):
        super(MergeSource, self).__init__(data_stream,
                                          data_stream.produces_examples,
                                          *args,
                                          **kwargs)

        self.merge_sources = merge_sources
        self.merge_name = merge_name
        self.sources = (merge_name,) + tuple(s for s in data_stream.sources if s not in merge_sources)

    def transform_any(self, data):
        merged_data = {s: d for s, d in zip(self.data_stream.sources, data)
                       if s in self.merge_sources}
        return [merged_data] + [d for d, s in zip(data, self.data_stream.sources)
                                if s not in self.merge_sources]
