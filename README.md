A simple python implementation of byte pair encoding.

I opted for being functional rather than object oriented because it's honestly a lot easier to understand this way. Some of the pretokenization code is based on [cs336](https://stanford-cs336.github.io/spring2025/) from Stanford, which is a great resource if you want to learn about LLMs.

To run, use:
```
python bpe.py <filename> <vocab_size> <num_processors> <special_tokens>
```

For testing, you can use:
```
python bpe.py test.txt
```