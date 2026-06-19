
### CHAIN ###

from collections import namedtuple
from functools import partial
from operator import itemgetter
from threading import Semaphore, Thread

def passThrough(x: Any, *args) -> Any:
    return x

def formatStr(template: str, x: dict) -> str:
    return template.format(**x)

def pushResult(results: dict, lock: Lock, key: str, func: Callable, x: Any) -> None:
    with lock:
        results[key] = func(x)

def callThread(kfxs: list[tuple[str,Callable,Any]]) -> dict:
    args = ({}, Semaphore(4))
    ts = [ Thread(target=pushResult, args=args + kfx) for kfx in kfxs ]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return args[0]

def callList(callables: list[Callable], x: Any) -> list:
    return [ f(x) for f in callables ]

def chooseList(callables: list[Callable], x: Any) -> Any:
    for f in callables:
        y = f(x)
        if y is not None:
            return y

def callDict(callabledict: dict[str,Callable], x: Any) -> dict:
    return { k: f(x) for k,f in callabledict.items() }

def assignDict(callabledict: dict[str,Callable], x: dict) -> dict:
    y = callDict(callabledict, x)
    for k in set(xdict) - set(y):
        y[k] = x[k]
    return y

def makeCallable(x: Any, call_dict=callDict, call_list=callList) -> Callable:
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
    def __call__(self, x: Any) -> Any:
        for f in self.callables:
            x = f(x)
        return x
    def __or__(self, other: Any) -> Chain:
        return Chain(self, other)
    def __ror__(self, other: Any) -> Chain:
        return Chain(other, self)

def buildAssign(*args, **kwds) -> Chain:
    kwds.update(zip(args[::2], args[1::2]))
    return Chain(makeCallable(kwds, call_dict=assignDict))

def buildChoose(*args) -> Chain:
    return Chain(makeCallable(args, call_list=chooseList))

def buildPick(*args) -> Chain:
    return Chain(itemgetter(*args))

def mapThread(f: Callable, xs: list) -> Iterator:
    buf = callThread( (i, f, x) for i,x in enumerate(xs) )
    return map(buf.get, range(len(buf)))

def buildMap(f: Callable, mapper=mapThread) -> Chain:
    return Chain(partial(mapper or map, f))

### RAG ###

def buildCondense(llm: Callable, prompt: str=None, tag: str=None) -> Chain:
    condense = Chain(
prompt or '''Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History: ```
{chat_history}
```
Follow Up Question: {question}
Standalone question:''', llm)
    f = lambda x: condense(x) if x.get('chat_history') else x['question']
    return Chain({tag: f} if tag else f)

def buildRetrieve(retriever: Callable, pick: str=None, lift='question', tag='context') -> Chain:
    if pick:
        return buildAssign(tag, buildPick(pick) | retriever)
    return Chain({lift: passThrough, tag: retriever})

def buildAnswer(llm: Callable, prompt=None, join='context', tag='answer'):
    answer = Chain(
prompt or '''Answer the question in its original language, based only on the following context: ```
{context}
```
Question: {question}
Answer:''', llm)
    if join:
        f = lambda xs: '\n\n'.join( x.page_content for x in xs[join] )
        answer = buildAssign(join, f) | answer
    return buildAssign(tag, answer)

def buildRAG(retriever: Callable, llm_fast: Callable, llm: Callable) -> Chain:
    return buildCondense(llm_fast) | buildRetrieve(retriever) | buildAnswer(llm)

### RETRIEVE ###

Document = namedtuple('Document', ['page_content', 'metadata'])

def makeDocument(item: dict) -> Document:
    md = { k: item[k] for k in set(item) - {'text', 'vector'} }
    return Document(page_content=item['text'], metadata=md)

def docDistance(document: Document) -> float:
    return document.metadata['_distance']

### LLM ###

def callBedrock(converse: Callable, model_id: str, max_tokens: int, text: str) -> str:
    conf = dict(inferenceConfig=dict(temperature=0, maxTokens=max_tokens))
    if isinstance(text, (list, tuple)):
        text, conf['system'] = text[-1], [dict(text=text[0])]
    conf['messages'] = [dict(content=[dict(text=text)], role='user')]
    res = converse(modelId=model_id, **conf)
    return res['output']['message']['content'][0]['text']

def makeBedrockLLM(bedrock, *args) -> Callable | Iterator[Callable]:
    fs = ( partial(callBedrock, bedrock.converse, *xs) for xs in args )
    return next(fs) if len(args) == 1 else fs

if __name__ == '__main__':
    retrieve = lambda text: [Document('test ' + text, {})]
    condense = lambda text: 'hello with history'
    answer = lambda text: 'hey!'
    rag = buildRAG(retrieve, condense, answer)
    print(rag(dict(question='hello')))
    print(rag(dict(question='hello', chat_history=['some history'])))


