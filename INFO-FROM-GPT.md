
# SRL

**QA‑SRL Bank 2.0** нужен для обучения модели находить у каждого глагольного события его участников и обстоятельства: кто совершил действие, над чем, кому, где, когда, почему, каким способом и т. д.

Это не обычный вопросно-ответный датасет и не диалоговый корпус. Вопросы здесь служат человекочитаемыми семантическими ролями.

```text
Sentence:
John gave Mary a book in Paris.

Predicate:
gave

Who gave someone something?       → John
Who did someone give something to? → Mary
What did someone give someone?     → a book
Where did someone give something?  → in Paris
```

В QA‑SRL Bank 2.0 содержится 64 018 предложений, 133 479 размеченных глагольных предикатов и 265 140 валидированных пар «вопрос — ответ». 
**Источники**: Wikipedia, Wikinews и научно-учебные тексты TQA. **>>** [Large-Scale QA-SRL Parsing](https://arxiv.org/html/1805.05377v1)



## Что можно обучать на этом датасете

Основные варианты:

1. Поиск аргументов глагола
```
Вход:
Sentence: John gave Mary a book.
Predicate: gave

Выход:
John
Mary
a book
```

2. Генерация семантического вопроса для аргумента
```
Вход:
Sentence: John gave Mary a book.
Predicate: gave
Answer: Mary

Выход:
Who did someone give something to?
```

3. Полный QA‑SRL parsing
```
Вход:
Sentence: John gave Mary a book.
Predicate: gave

Выход:
[
  {
    "question": "Who gave someone something?",
    "answers": ["John"]
  },
  {
    "question": "Who did someone give something to?",
    "answers": ["Mary"]
  },
  {
    "question": "What did someone give someone?",
    "answers": ["a book"]
  }
]
```

4. Преобразование в event extraction
Для вашей схемы извлечения событий QA‑SRL особенно полезен:
```
{
  "action_text": "gave",
  "action_lemma": "give",
  "arguments": [
    {
      "question": "Who gave someone something?",
      "answer": "John"
    },
    {
      "question": "Who did someone give something to?",
      "answer": "Mary"
    },
    {
      "question": "What did someone give someone?",
      "answer": "a book"
    }
  ]
}
```

Но важно: корпус не содержит готовых меток `agent`, `patient`, `recipient`. Семантическое отношение выражено естественным вопросом.


#### Как скачать на Windows

Самый простой вариант — официальный архив, без Hugging Face.
```bash
curl.exe -L "https://qasrl.org/data/qasrl-v2.tar" -o "qasrl-v2.tar"
```

Официальный архив: https://qasrl.org/data/qasrl-v2.tar

Альтернативный вариант через Git:
```bash
git clone https://github.com/uwnlp/qasrl-bank.git
cd qasrl-bank
bash download.sh
```

Старый репозиторий теперь указывает на более общий проект `julianmichael/qasrl`, однако он всё ещё содержит корректное описание Bank 2.0 и официальный загрузчик.


#### Структура скачанного датасета

Ожидаемая структура:
```text
qasrl-v2/
├── orig/
│   ├── train.jsonl.gz
│   ├── dev.jsonl.gz
│   └── test.jsonl.gz
│
├── expanded/
│   ├── train.jsonl.gz
│   └── dev.jsonl.gz
│
├── dense/
│   ├── dev.jsonl.gz
│   └── test.jsonl.gz
│
└── index.json.gz
```

Назначение каталогов:
Каталог	Что внутри	Как использовать
orig	Изначальные вопросы, написанные разметчиками	Чистое обучение и базовая оценка
expanded	Расширенные данные с дополнительными модельными вопросами, проверенными людьми	Предпочтительно для обучения
dense	Более плотная разметка части dev/test, обычно 6 суждений на вопрос	Строгая оценка
index.json.gz	Документы, домены и заголовки	Обычно для обучения не нужен

* Официальная рекомендация — использовать `orig` или `expanded` для обучения, а `orig` и `dense` для оценки. GitHub
* Не объединяйте `orig/train` и `expanded/train` простым сложением: `expanded` может уже содержать исходные и дополнительные аннотации. Иначе получите дубли.


#### Как устроена одна запись
Файлы имеют формат `JSON Lines`, сжатый через gzip. Одна строка соответствует одному предложению и содержит все размеченные глаголы этого предложения. >> https://github.com/uwnlp/qasrl-bank/blob/master/FORMAT.md


Упрощённо это выглядит так:
```json
{
  "sentenceId": "wikipedia:123:4",
  "sentenceTokens": [
    "John",
    "gave",
    "Mary",
    "a",
    "book",
    "."
  ],
  "verbEntries": {
    "1": {
      "verbIndex": 1,
      "verbInflectedForms": {
        "stem": "give",
        "presentSingular3rd": "gives",
        "presentParticiple": "giving",
        "past": "gave",
        "pastParticiple": "given"
      },
      "questionLabels": {
        "Who gave someone something?": {
          "questionString": "Who gave someone something?",
          "questionSlots": {
            "wh": "who",
            "aux": "_",
            "subj": "_",
            "verb": "past",
            "obj": "someone",
            "prep": "_",
            "obj2": "something"
          },
          "tense": "past",
          "isPerfect": false,
          "isProgressive": false,
          "isNegated": false,
          "isPassive": false,
          "answerJudgments": [
            {
              "sourceId": "turk-...",
              "isValid": true,
              "spans": [[0, 1]]
            }
          ]
        }
      }
    }
  }
}
```

`spans` — позиции ответа в `sentenceTokens`:
```text
[start, end)
```

Правая граница не включается:
```python
tokens[start:end]
```

Например:
```python
tokens = ["John", "gave", "Mary", "a", "book", "."]
span = [3, 5]

tokens[3:5]
# ["a", "book"]
```


#### Чтение `.jsonl.gz` напрямую

Распаковывать каждый файл необязательно:
```python
import gzip
import json
from pathlib import Path


def read_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


path = Path("qasrl-v2/expanded/train.jsonl.gz")

for sentence in read_jsonl_gz(path):
    print(sentence["sentenceId"])
    print(sentence["sentenceTokens"])
    print(sentence["verbEntries"])
    break
```

#### Преобразование в плоский JSONL

Скрипт в текущем репозитории, который создаёт одну запись на один валидный вопрос: `qasrl_v2.py`

Для `orig` и `expanded` разумный строгий вариант:
```python
min_valid_ratio=1.0
```

То есть все имеющиеся разметчики должны признать вопрос валидным.
Для dense можно использовать критерий 5 из 6:
```python
min_valid_ratio=5 / 6
```

Именно правило «5 из 6 считают вопрос валидным» применялось в строгой финальной оценке авторов: https://arxiv.org/html/1805.05377v1


### Как готовить данные для вашей decoder-only модели
Для вашей модели я бы делал не обычные независимые QA-примеры, а одну запись на предикат.

* Вход:
```text
<task>qasrl</task>
<sentence>John gave Mary a book in Paris.</sentence>
<predicate index="1">gave</predicate>
<output>
```

* Цель:
```json
{
  "predicate": "gave",
  "lemma": "give",
  "arguments": [
    {
      "question": "Who gave someone something?",
      "answers": [
        {"start": 0, "end": 1, "text": "John"}
      ]
    },
    {
      "question": "Who did someone give something to?",
      "answers": [
        {"start": 2, "end": 3, "text": "Mary"}
      ]
    }
  ]
}
```

* При causal-LM обучении:
```text
input/prompts tokens → labels = -100
target JSON tokens   → обычные token IDs
```

То есть loss считается только по целевому JSON, а предложение и указанный predicate являются контекстом.


### Какая стратегия обучения лучше
Для небольшой GPT-подобной модели оптимально разделить задачу на этапы:

* Span detection
```text
sentence + predicate → список аргументных spans
```

* Question generation
```text
sentence + predicate + answer span → QA-SRL question
```

* После этого — совместная задача:
```text
sentence + predicate → полный список question-answer pairs
```

Так была концептуально устроена и исходная система: сначала обнаружение аргументных spans, затем генерация вопроса, описывающего отношение аргумента к глаголу.


Для вашей event-extraction модели я бы начал с:
```text
expanded/train → обучение
dense/dev      → подбор параметров
dense/test     → окончательная оценка
```

