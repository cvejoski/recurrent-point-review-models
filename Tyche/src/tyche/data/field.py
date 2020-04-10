import torch
from torchtext.data import Field, NestedField, Dataset


class BPTTField(Field):
    def __init__(self, bptt_length=20, **kwargs):
        super(BPTTField, self).__init__(sequential=True, batch_first=True, **kwargs)
        self._bptt_length = bptt_length

    def pad(self, minibatch):
        """Pad a batch of examples using this field.

        Pads to self.fix_length if provided, otherwise pads to the length of
        the longest example in the batch. Prepends self.init_token and appends
        self.eos_token if those attributes are not None. Returns a tuple of the
        padded list and a list containing lengths of each example if
        `self.include_lengths` is `True` and `self.sequential` is `True`, else just
        returns the padded list. If `self.sequential` is `False`, no padding is applied.
        """
        minibatch = list(minibatch)
        if not self.sequential:
            return minibatch

        if self.fix_length is None:
            max_len = max(len(x) for x in minibatch)
        else:
            max_len = self.fix_length + (
                self.init_token, self.eos_token).count(None) - 2

        reminder = max_len % self._bptt_length
        if reminder != 0:
            max_len += (self._bptt_length - reminder)
        padded, lengths = [], []
        for x in minibatch:
            if self.pad_first:
                padded.append(
                        [self.pad_token] * max(0, max_len - len(x)) +
                        ([] if self.init_token is None else [self.init_token]) +
                        list(x[-max_len:] if self.truncate_first else x[:max_len]) +
                        ([] if self.eos_token is None else [self.eos_token]))
            else:
                padded.append(
                        ([] if self.init_token is None else [self.init_token]) +
                        list(x[-max_len:] if self.truncate_first else x[:max_len]) +
                        ([] if self.eos_token is None else [self.eos_token]) +
                        [self.pad_token] * max(0, max_len - len(x)))
            lengths.append(len(padded[-1]) - max(0, max_len - len(x)))
        if self.include_lengths:
            return (padded, lengths)
        return padded

    def numericalize(self, arr, device=None):
        r = super(BPTTField, self).numericalize(arr, device)
        if self.include_lengths:
            return r[0], r[1].long()
        return r


class NestedBPTTField(BPTTField):
    """A nested field.

    A nested field holds another field (called *nesting field*), accepts an untokenized
    string or a list string tokens and groups and treats them as one field as described
    by the nesting field. Every token will be preprocessed, padded, etc. in the manner
    specified by the nesting field. Note that this means a nested field always has
    ``sequential=True``. The two fields' vocabularies will be shared. Their
    numericalization results will be stacked into a single tensor. And NestedField will
    share the same include_lengths with nesting_field, so one shouldn't specify the
    include_lengths in the nesting_field. This field is
    primarily used to implement character embeddings. See ``tests/data/test_field.py``
    for examples on how to use this field.

    Arguments:
        nesting_field (Field): A field contained in this nested field.
        use_vocab (bool): Whether to use a Vocab object. If False, the data in this
            field should already be numerical. Default: ``True``.
        init_token (str): A token that will be prepended to every example using this
            field, or None for no initial token. Default: ``None``.
        eos_token (str): A token that will be appended to every example using this
            field, or None for no end-of-sentence token. Default: ``None``.
        fix_length (int): A fixed length that all examples using this field will be
            padded to, or ``None`` for flexible sequence lengths. Default: ``None``.
        dtype: The torch.dtype class that represents a batch of examples
            of this kind of data. Default: ``torch.long``.
        preprocessing (Pipeline): The Pipeline that will be applied to examples
            using this field after tokenizing but before numericalizing. Many
            Datasets replace this attribute with a custom preprocessor.
            Default: ``None``.
        postprocessing (Pipeline): A Pipeline that will be applied to examples using
            this field after numericalizing but before the numbers are turned
            into a Tensor. The pipeline function takes the batch as a list, and
            the field's Vocab. Default: ``None``.
        include_lengths: Whether to return a tuple of a padded minibatch and
            a list containing the lengths of each examples, or just a padded
            minibatch. Default: False.
        tokenize (callable or str): The function used to tokenize strings using this
            field into sequential examples. If "spacy", the SpaCy English tokenizer is
            used. Default: ``lambda s: s.split()``
        pad_token (str): The string token used as padding. If ``nesting_field`` is
            sequential, this will be set to its ``pad_token``. Default: ``"<pad>"``.
        pad_first (bool): Do the padding of the sequence at the beginning. Default:
            ``False``.
    """

    def __init__(self, nesting_field, bptt_length=50, use_vocab=True, init_token=None, eos_token=None,
                 fix_length=None, dtype=torch.long, preprocessing=None,
                 postprocessing=None, tokenize=lambda s: s.split(),
                 include_lengths=False, pad_token='<pad>',
                 pad_first=False, truncate_first=False):

        if isinstance(nesting_field, NestedField):
            raise ValueError('nesting field must not be another NestedField')
        if nesting_field.include_lengths:
            raise ValueError('nesting field cannot have include_lengths=True')

        if nesting_field.sequential:
            pad_token = nesting_field.pad_token
        super(NestedBPTTField, self).__init__(
                use_vocab=use_vocab,
                init_token=init_token,
                eos_token=eos_token,
                fix_length=fix_length,
                dtype=dtype,
                preprocessing=preprocessing,
                postprocessing=postprocessing,
                lower=nesting_field.lower,
                tokenize=tokenize,
                pad_token=pad_token,
                unk_token=nesting_field.unk_token,
                pad_first=pad_first,
                truncate_first=truncate_first,
                include_lengths=include_lengths,
                bptt_length=bptt_length
        )
        self.nesting_field = nesting_field
        # in case the user forget to do that
        self.nesting_field.batch_first = True

    def preprocess(self, xs):
        """Preprocess a single example.

        Firstly, tokenization and the supplied preprocessing pipeline is applied. Since
        this field is always sequential, the result is a list. Then, each element of
        the list is preprocessed using ``self.nesting_field.preprocess`` and the resulting
        list is returned.

        Arguments:
            xs (list or str): The input to preprocess.

        Returns:
            list: The preprocessed list.
        """
        return [self.nesting_field.preprocess(x)
                for x in super(NestedBPTTField, self).preprocess(xs)]

    def pad(self, minibatch):
        """Pad a batch of examples using this field.

        If ``self.nesting_field.sequential`` is ``False``, each example in the batch must
        be a list of string tokens, and pads them as if by a ``Field`` with
        ``sequential=True``. Otherwise, each example must be a list of list of tokens.
        Using ``self.nesting_field``, pads the list of tokens to
        ``self.nesting_field.fix_length`` if provided, or otherwise to the length of the
        longest list of tokens in the batch. Next, using this field, pads the result by
        filling short examples with ``self.nesting_field.pad_token``.

        Example:
            >>> import pprint
            >>> pp = pprint.PrettyPrinter(indent=4)
            >>>
            >>> nesting_field = Field(pad_token='<c>', init_token='<w>', eos_token='</w>')
            >>> field = NestedField(nesting_field, init_token='<s>', eos_token='</s>')
            >>> minibatch = [
            ...     [list('john'), list('loves'), list('mary')],
            ...     [list('mary'), list('cries')],
            ... ]
            >>> padded = field.pad(minibatch)
            >>> pp.pprint(padded)
            [   [   ['<w>', '<s>', '</w>', '<c>', '<c>', '<c>', '<c>'],
                    ['<w>', 'j', 'o', 'h', 'n', '</w>', '<c>'],
                    ['<w>', 'l', 'o', 'v', 'e', 's', '</w>'],
                    ['<w>', 'm', 'a', 'r', 'y', '</w>', '<c>'],
                    ['<w>', '</s>', '</w>', '<c>', '<c>', '<c>', '<c>']],
                [   ['<w>', '<s>', '</w>', '<c>', '<c>', '<c>', '<c>'],
                    ['<w>', 'm', 'a', 'r', 'y', '</w>', '<c>'],
                    ['<w>', 'c', 'r', 'i', 'e', 's', '</w>'],
                    ['<w>', '</s>', '</w>', '<c>', '<c>', '<c>', '<c>'],
                    ['<c>', '<c>', '<c>', '<c>', '<c>', '<c>', '<c>']]]

        Arguments:
            minibatch (list): Each element is a list of string if
                ``self.nesting_field.sequential`` is ``False``, a list of list of string
                otherwise.

        Returns:
            list: The padded minibatch. or (padded, sentence_lens, word_lengths)
        """
        minibatch = list(minibatch)
        if not self.nesting_field.sequential:
            return super(NestedBPTTField, self).pad(minibatch)

        # Save values of attributes to be monkeypatched
        old_pad_token = self.pad_token
        old_init_token = self.init_token
        old_eos_token = self.eos_token
        old_fix_len = self.nesting_field.fix_length
        # Monkeypatch the attributes
        if self.nesting_field.fix_length is None:
            max_len = max(len(xs) for ex in minibatch for xs in ex)
            fix_len = max_len + 2 - (self.nesting_field.init_token,
                                     self.nesting_field.eos_token).count(None)
            self.nesting_field.fix_length = fix_len
        self.pad_token = [self.pad_token] * self.nesting_field.fix_length
        if self.init_token is not None:
            # self.init_token = self.nesting_field.pad([[self.init_token]])[0]
            self.init_token = [self.init_token]
        if self.eos_token is not None:
            # self.eos_token = self.nesting_field.pad([[self.eos_token]])[0]
            self.eos_token = [self.eos_token]
        # Do padding
        old_include_lengths = self.include_lengths
        self.include_lengths = True
        self.nesting_field.include_lengths = True
        padded, sentence_lengths = super(NestedBPTTField, self).pad(minibatch)
        padded_with_lengths = [self.nesting_field.pad(ex) for ex in padded]
        word_lengths = []
        final_padded = []
        max_sen_len = len(padded[0])

        for (pad, lens), sentence_len in zip(padded_with_lengths, sentence_lengths):
            if sentence_len == max_sen_len:
                lens = lens
                pad = pad
            elif self.pad_first:
                lens[:(max_sen_len - sentence_len)] = (
                        [0] * (max_sen_len - sentence_len))
                pad[:(max_sen_len - sentence_len)] = (
                        [self.pad_token] * (max_sen_len - sentence_len))
            else:
                lens[-(max_sen_len - sentence_len):] = (
                        [0] * (max_sen_len - sentence_len))
                pad[-(max_sen_len - sentence_len):] = (
                        [self.pad_token] * (max_sen_len - sentence_len))
            word_lengths.append(lens)
            final_padded.append(pad)
        padded = final_padded

        # Restore monkeypatched attributes
        self.nesting_field.fix_length = old_fix_len
        self.pad_token = old_pad_token
        self.init_token = old_init_token
        self.eos_token = old_eos_token
        self.include_lengths = old_include_lengths
        if self.include_lengths:
            return padded, sentence_lengths, word_lengths
        return padded

    def build_vocab(self, *args, **kwargs):
        """Construct the Vocab object for nesting field and combine it with this field's vocab.

        Arguments:
            Positional arguments: Dataset objects or other iterable data
                sources from which to construct the Vocab object that
                represents the set of possible values for the nesting field. If
                a Dataset object is provided, all columns corresponding
                to this field are used; individual columns can also be
                provided directly.
            Remaining keyword arguments: Passed to the constructor of Vocab.
        """
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                sources.extend(
                        [getattr(arg, name) for name, field in arg.fields.items()
                         if field is self]
                )
            else:
                sources.append(arg)

        flattened = []
        for source in sources:
            flattened.extend(source)
        old_vectors = None
        old_unk_init = None
        old_vectors_cache = None
        if "vectors" in kwargs.keys():
            old_vectors = kwargs["vectors"]
            kwargs["vectors"] = None
        if "unk_init" in kwargs.keys():
            old_unk_init = kwargs["unk_init"]
            kwargs["unk_init"] = None
        if "vectors_cache" in kwargs.keys():
            old_vectors_cache = kwargs["vectors_cache"]
            kwargs["vectors_cache"] = None
        # just build vocab and does not load vector
        self.nesting_field.build_vocab(*flattened, **kwargs)
        super(NestedBPTTField, self).build_vocab()
        self.vocab.extend(self.nesting_field.vocab)
        self.vocab.freqs = self.nesting_field.vocab.freqs.copy()
        if old_vectors is not None:
            self.vocab.load_vectors(old_vectors,
                                    unk_init=old_unk_init, cache=old_vectors_cache)

        self.nesting_field.vocab = self.vocab

    def numericalize(self, arrs, device=None):
        """Convert a padded minibatch into a variable tensor.

        Each item in the minibatch will be numericalized independently and the resulting
        tensors will be stacked at the first dimension.

        Arguments:
            arr (List[List[str]]): List of tokenized and padded examples.
            device (str or torch.device): A string or instance of `torch.device`
                specifying which device the Variables are going to be created on.
                If left as default, the tensors will be created on cpu. Default: None.
        """
        numericalized = []
        self.nesting_field.include_lengths = False
        if self.include_lengths:
            arrs, sentence_lengths, word_lengths = arrs

        for arr in arrs:
            numericalized_ex = self.nesting_field.numericalize(
                    arr, device=device)
            numericalized.append(numericalized_ex)
        padded_batch = torch.stack(numericalized)

        self.nesting_field.include_lengths = True
        if self.include_lengths:
            sentence_lengths = \
                torch.tensor(sentence_lengths, dtype=torch.int64, device=device)
            word_lengths = torch.tensor(word_lengths, dtype=torch.int64, device=device)
            return (padded_batch, sentence_lengths, word_lengths)
        return padded_batch


class ReversibleField(Field):
    def __init__(self, **kwargs):
        if 'unk_token' not in kwargs:
            kwargs['unk_token'] = '<unk>'
        super(ReversibleField, self).__init__(**kwargs)

    def reverse(self, batch):

        if not self.batch_first:
            batch = batch.t()
        with torch.cuda.device_of(batch):
            batch = batch.tolist()
        batch = [[self.vocab.itos[ind] for ind in ex] for ex in batch]  # denumericalize

        def trim(s, t):
            sentence = []
            for w in s:
                if w == t:
                    break
                sentence.append(w)
            return sentence

        batch = [trim(ex, self.eos_token) for ex in batch]  # trim past frst eos

        def filter_special(tok):
            return tok not in (self.init_token, self.pad_token)

        batch = [filter(filter_special, ex) for ex in batch]

        return [' '.join(ex) for ex in batch]
