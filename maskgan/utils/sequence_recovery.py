import torch


class SequenceGenerator:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, _input):

        return self.generate(_input)

    def generate(self, tensor):
        ids = tensor.tolist()
        texts = []
        for id in ids:
            text = self.ids2string(id, rem_eos=True)
            texts.append(text)

        return texts

    def id2char(self, id):
        i2c = {v: k for k, v in self.vocab.items()}

        return i2c[id]

    def ids2string(self, ids, rem_eos=True):

        if len(ids) == 0:

            return ''

        if rem_eos and ids[-1] == self.vocab['<eos>']:
            ids = ids[:-1]

        string = ''.join(self.id2char(id) for id in ids)

        return string


def pretty_print(vocab, srcs, tgts, generated, truncate=None):
    sequence_generator = SequenceGenerator(vocab)

    srcs = sequence_generator(srcs)
    tgts = sequence_generator(tgts)
    generated = sequence_generator(generated)

    lines = []
    truncate = truncate if truncate is not None else len(srcs)

    for _srcs, _tgts, _generated in zip(srcs, tgts, generated):
        lines.append(f'> {_srcs}')
        lines.append(f'< {_tgts}')
        lines.append(f'= {_generated}')
        lines.append("")

        truncate -= 1
        if truncate <= 0:
            break

    # logger('<br>'.join(lines))
