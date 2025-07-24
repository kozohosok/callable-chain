
### CHAIN ###

from collections import namedtuple
from functools import partial
from operator import itemgetter

def passThrough(x, *args):
    return x

def formatStr(template, xdict):
    return template.format(**xdict)

def callList(callables, x):
    return [ f(x) for f in callables ]

def chooseList(callables, x):
    for f in callables:
        y = f(x)
        if y is not None:
            return y

def callDict(callabledict, x):
    return { k: f(x) for k,f in callabledict.items() }

def assignDict(callabledict, xdict):
    y = callDict(callabledict, xdict)
    for k in set(xdict) - set(y):
        y[k] = xdict[k]
    return y

def makeCallable(x, call_dict=callDict, call_list=callList):
    if callable(x):
        return x
    if isinstance(x, str):
        return partial(formatStr, x)
    if isinstance(x, dict):
        return partial(call_dict, { k: makeCallable(v) for k,v in x.items() })
    if isinstance(x, (list, tuple)):
        return partial(call_list, list(map(makeCallable, x)))
    return partial(passThrough, x)

class Chain:
    def __init__(self, *args):
        fs = []
        for x in args:
            fs += x.callables if isinstance(x, Chain) else [makeCallable(x)]
        self.callables = fs
    def __call__(self, x):
        for f in self.callables:
            x = f(x)
        return x
    def __or__(self, other):
        return Chain(self, other)
    def __ror__(self, other):
        return Chain(other, self)

def buildAssign(*args, **kwds):
    kwds.update(zip(args[::2], args[1::2]))
    return Chain(makeCallable(kwds, call_dict=assignDict))

def buildChoose(*args):
    return Chain(makeCallable(args, call_list=chooseList))

def buildPick(*args, **kwds):
    return Chain(itemgetter(*args, **kwds))

def buildMap(f):
    return Chain(partial(map, f))

### RAG ###

def buildCondense(llm, prompt=None, tag=None):
    condense = Chain(
prompt or '''Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History: ```
{chat_history}
```
Follow Up QUestion: {question}
Standalone question:''', llm)
    f = lambda x: condense(x) if x.get('chat_history') else x['question']
    return Chain({tag: f} if tag else f)

def buildRetrieve(retriever, pick=None, lift='question', tag='context'):
    if pick:
        return buildAssign(tag, buildPick(pick) | retriever)
    return Chain({lift: passThrough, tag: retriever})

def buildAnswer(llm, prompt=None, join='context', tag='answer'):
    answer = Chain(
prompt or '''Answer the question in its own language, based only on the following context: ```
{context}
```
Question: {question}
Answer:''', llm)
    if join:
        f = lambda xs: '\n\n'.join( x.page_content for x in xs[join] )
        answer = buildAssign(join, f) | answer
    return buildAssign(tag, answer)

def buildRAG(retriever, llm_fast, llm):
    return buildCondense(llm_fast) | buildRetrieve(retriever) | buildAnswer(llm)

### RETRIEVE ###

Document = namedtuple('Document', ['page_content', 'metadata'])

def makeDocument(item):
    md = { k: item[k] for k in set(item) - {'text', 'vector'} }
    return Document(page_content=item['text'], metadata=md)

def docDistance(document):
    return document.metadata['_distance']

if __name__ == '__main__':
    retrieve = lambda text: [Document('test', {})]
    answer = lambda text: 'hey!'
    condense = lambda text: 'hello with history'
    rag = buildRAG(retrieve, condense, answer)
    print(rag(dict(question='hello')))
    print(rag(dict(question='hello', chat_history='some history')))
