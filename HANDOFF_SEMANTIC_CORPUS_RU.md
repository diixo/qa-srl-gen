# Python QA-SRL и генератор семантического обучающего корпуса

## Назначение документа

Этот документ является техническим handoff для Codex. Его нужно прочитать полностью перед началом разработки. Он фиксирует цели проекта, уже принятые архитектурные решения, онтологию, ограничения, этапы реализации и критерии готовности.

Главная цель — создать на Python расширяемую систему, которая:

1. переносит полезное ядро проекта QA-SRL из Scala в Python;
2. анализирует реальные входные тексты и строит их семантическое представление;
3. генерирует контролируемые синтетические ситуации посредством подстановки действий, имён, сущностей, состояний и свойств;
4. отдельно генерирует естественные вопросы и ответы по уже известной семантической структуре;
5. создаёт корпус для обучения небольшой decoder-only модели пониманию текста и ответам на вопросы;
6. позволяет постепенно добавлять новые входные тексты, версии онтологии и новые типы генерации.

Модель не обязана отвечать в JSON. Целевой ответ модели должен быть обычным естественным текстом. Структурированный формат нужен только внутри генератора для контроля правильности и получения различных экспортов.

## Исходные материалы

### Основной репозиторий

- Репозиторий: <https://github.com/julianmichael/qasrl>
- Лицензия кода: MIT. При адаптации исходного кода необходимо сохранить исходное уведомление об авторских правах и указать, какие части были перенесены или переработаны.
- Репозиторий содержит Scala/Mill-модули для QA-SRL, QA-SRL Bank, crowdsourcing, browser, services и вспомогательные приложения.
- Полный буквальный перенос всего репозитория не нужен.

### Наборы данных

Предусмотреть поддержку:

- QA-SRL Bank 2.1 — основной источник для обучения на глагольных предикатах;
- QA-SRL Bank 2.0 — совместимость с исходным форматом;
- QA-SRL GS — качественная контрольная выборка;
- QANom — предикатно-аргументная разметка номинализаций;
- произвольные пользовательские TXT и JSONL;
- диалоговые корпуса наподобие DailyDialog.

### Архив Wiktionary

Пользователь располагает файлом `wiktionary.tar.gz`, загруженным из репозитория QA-SRL. Это не QA-SRL-корпус и не NER-датасет. Это вспомогательный морфологический ресурс.

Проверенное содержимое архива:

| Файл | Примерное число строк | Назначение |
|---|---:|---|
| `en_verb_inflections.txt` | 21 970 | Пять форм английского глагола |
| `en_postags_withverb.txt` | 29 449 | Части речи для слов, имеющих глагольное употребление |
| `verb_phrases.txt` | 2 085 | Многословные глагольные выражения |
| `extract_english_verb_inflections.py` | 128 | Старый скрипт извлечения форм |
| `extract_english_all_postags.py` | 37 | Старый скрипт извлечения частей речи |
| `extract_english_verb_particles.py` | 40 | Старый скрипт извлечения глагольных выражений |

SHA-256 проверенного архива:

```text
3dbfe046fe0dffbb91c5df3fe513589998655a7032a91a4125a6b8db71314ada
```

Формат `en_verb_inflections.txt` — TSV с пятью полями:

```text
lemma    present_singular_3rd    present_participle    simple_past    past_participle
give     gives                   giving                gave           given
go       goes                    going                 went           gone
```

Ограничения ресурса:

- это старый scrape Wiktionary;
- скрипты написаны под Python 2 и старую структуру шаблонов Wiktionary;
- словарь содержит редкие, диалектные и неоднозначные формы;
- некоторые слова одновременно являются существительными, именами или глаголами;
- многословные и дефисные формы покрыты неполно;
- `be` и некоторые специальные парадигмы отсутствуют или должны обрабатываться отдельно;
- нахождение слова в словаре не является достаточным основанием присвоить ему `ACTION`.

Словарь должен только предлагать кандидатов и восстанавливать парадигму. Окончательное семантическое решение принимается по контексту.

## Ключевое архитектурное разделение

Нельзя смешивать четыре разные задачи в один непрозрачный генератор.

### 1. Python-ядро QA-SRL

Работает с форматом QA-SRL, вопросными слотами, глагольными формами, spans и проверкой допустимости вопросов.

### 2. Семантический генератор

Создаёт контролируемые синтетические ситуации из онтологии, action/state frames и типизированных слотов.

### 3. Семантический разметчик

Получает реальный входной текст или диалог и строит внутреннее семантическое представление.

### 4. Генератор вопросов

Получает текст и уже проверенную семантическую структуру, после чего создаёт естественные вопросы и ответы. Он не должен самостоятельно придумывать ground truth.

Общий поток:

```text
онтология + frames ──> генератор ситуаций ──┐
                                           ├──> семантическая запись ──> генератор вопросов
реальные тексты ─────> разметчик ──────────┘
```

Такая схема позволяет из одной семантической записи получать:

- QA-корпус;
- NER-корпус;
- SRL-корпус;
- корпус `ACTION/STATE`;
- корпус свойств;
- корпус эмоций;
- корпус speech acts и dialogue acts.

## Онтология версии 1

Метки не являются полностью взаимоисключающими. Один span может иметь несколько меток. Не использовать единственную плоскую BIO-схему как каноническое хранилище.

### Предикаты

#### ACTION

Динамическое событие, действие, процесс или изменение:

```text
run, give, open, build, travel, break, become
```

#### STATE

Состояние, владение, знание, отношение, положение или существование:

```text
know, own, contain, remain, belong, exist
```

Правило для связки:

```text
Anna is tall.
```

- `tall` получает `STATE + PROPERTY`;
- `is` рассматривается как служебная связка и отдельно не получает `ACTION`.

Для прогрессивной конструкции:

```text
Anna is running.
```

- `running` получает `ACTION`;
- `is` является вспомогательным глаголом.

### Сущности

```text
PERSON
ANIMAL
ORGANIZATION
LOCATION
PHYSICAL_OBJECT
ABSTRACT_ENTITY
DATE
TIME
```

### Характеристики

#### PROPERTY

Описательное качество, условие или характеристика сущности:

```text
tall, slender, beautiful, red, large, broken, very old
```

Не каждое прилагательное автоматически является `PROPERTY`. Например, `former`, `alleged`, `main`, `other`, `medical` могут выражать время, модальность, отношение или классификацию. POS-тег создаёт только кандидата; контекстная модель принимает окончательное решение.

Для свойства хранить отношение:

```text
PROPERTY_OF
```

Желательно сохранять базовую характеристику и модификаторы:

```text
not very tall
head = tall
degree = very
negated = true
```

#### EMOTION

Эмоция является пересекающимся семантическим признаком и может сочетаться с другими метками:

```text
afraid  -> STATE + PROPERTY + EMOTION
joy     -> ABSTRACT_ENTITY + EMOTION
loves   -> STATE + EMOTION
```

### Именованность

```text
NAMED_ENTITY
NAMED_OBJECT
```

`NAMED_ENTITY` — дополнительная метка или атрибут, а не взаимоисключающий тип:

```text
Anna   -> PERSON + NAMED_ENTITY
Kyiv   -> LOCATION + NAMED_ENTITY
Acme   -> ORGANIZATION + NAMED_ENTITY
Rex    -> ANIMAL + NAMED_ENTITY, если контекст указывает на животное
```

`NAMED_OBJECT` применяется к индивидуально названным предметам, произведениям, аппаратам, судам, продуктам и моделям:

```text
Titanic -> PHYSICAL_OBJECT + NAMED_OBJECT + NAMED_ENTITY
```

### Дискурсивные маркеры

Добавить span-метку:

```text
DISCOURSE_MARKER
```

Формы маркеров:

```text
INTERJECTION
FILLED_PAUSE
BACKCHANNEL
RESPONSE_PARTICLE
EVALUATIVE_RESPONSE
```

Коммуникативные функции:

```text
REACTION
SURPRISE
HESITATION
THINKING
UNCERTAINTY
ACKNOWLEDGEMENT
AGREEMENT
ACCEPTANCE
CONFIRMATION
DISAGREEMENT
REJECTION
APPROVAL
DISAPPROVAL
```

Примеры:

| Форма | Тип | Возможная функция |
|---|---|---|
| `Oh` | `INTERJECTION` | реакция, осознание, удивление |
| `Wow` | `INTERJECTION` | реакция, удивление, восхищение |
| `Umm` | `FILLED_PAUSE` | колебание |
| `Hmm` | `BACKCHANNEL` или `FILLED_PAUSE` | размышление, сомнение, подтверждение слушания |
| `Okay` | `RESPONSE_PARTICLE` | принятие, подтверждение, согласие |
| `No` | `RESPONSE_PARTICLE` | отрицательный ответ, несогласие, отказ |
| `Good` | `EVALUATIVE_RESPONSE` | одобрение |

Контекст обязателен:

```text
Good.                  -> APPROVAL
It is a good car.      -> PROPERTY
I feel good.           -> STATE + PROPERTY, иногда EMOTION

Okay, I will do it.    -> ACKNOWLEDGEMENT + ACCEPTANCE
The result is okay.    -> PROPERTY

No.                    -> REJECTION или NEGATIVE_ANSWER
No cars arrived.       -> отрицательный квантификатор, не discourse marker
No problem.            -> REASSURANCE или ACCEPTANCE
```

### Аннотация всей реплики

Помимо span-меток, хранить атрибуты реплики:

```text
SPEECH_ACT
POLARITY
STANCE
MOOD
```

Основные speech acts:

```text
INFORM
QUESTION
ANSWER
REQUEST
COMMAND
AGREEMENT
DISAGREEMENT
CONFIRMATION
REJECTION
ACCEPTANCE
ACKNOWLEDGEMENT
APPROVAL
DISAPPROVAL
REACTION
HESITATION
UNCERTAINTY
BACKCHANNEL
GREETING
FAREWELL
THANKING
APOLOGY
REASSURANCE
```

Одна реплика может иметь несколько speech acts.

## Семантические отношения

Минимальный набор:

```text
AGENT_OF
PATIENT_OF
THEME_OF
RECIPIENT_OF
EXPERIENCER_OF
LOCATION_OF
TIME_OF
PROPERTY_OF
STATE_OF
MEMBER_OF
PART_OF
INSTANCE_OF
SUBTYPE_OF
```

QA-SRL-вопрос должен сохраняться как естественное описание связи, даже если дополнительно вычисляется нормализованная роль.

Пример:

```text
Anna gave her dog Rex a red ball in Kyiv.
```

Внутренняя структура:

```text
Anna     -> PERSON + NAMED_ENTITY
gave     -> ACTION, lemma=give
Rex      -> ANIMAL + NAMED_ENTITY
red      -> PROPERTY
ball     -> PHYSICAL_OBJECT
Kyiv     -> LOCATION + NAMED_ENTITY

Anna  --AGENT_OF------> gave
Rex   --RECIPIENT_OF--> gave
ball  --THEME_OF------> gave
Kyiv  --LOCATION_OF---> gave
red   --PROPERTY_OF---> ball
```

## Каноническое внутреннее представление

Формат модели не обязан быть JSON, но внутреннее представление должно быть структурированным и проверяемым. Реализовать Python-модели через `dataclasses` или Pydantic.

Минимальные объекты:

```text
Document
Passage
Utterance
Token
Span
EntityMention
Predicate
Property
Relation
DialogueAnnotation
AnnotationRun
```

Для каждого span хранить:

```text
id
exact_text
start_char
end_char
start_token
end_token
labels
head_token
normalized_form
confidence
source
review_status
```

Для предиката дополнительно:

```text
lemma
tense
aspect
voice
polarity
modality
```

Разрешить:

- вложенные spans;
- несколько меток на одном span;
- при необходимости разрывные spans для фразовых глаголов;
- несколько relations для одной сущности;
- несколько speech acts для одной реплики.

## Python-перенос QA-SRL

Нужен функциональный перенос полезного ядра, а не буквальное копирование всего Scala-проекта.

### Перенести в первую очередь

```text
qasrl_py/
├── models.py
├── bank_reader.py
├── inflections.py
├── question_slots.py
├── question_parser.py
├── question_renderer.py
├── state_machine.py
├── validation.py
└── autocomplete.py
```

Функциональные требования:

1. Потоково читать `jsonl.gz` без предварительной распаковки.
2. Читать QA-SRL Bank 2.0 и 2.1.
3. Представлять `Sentence`, `VerbEntry`, `QuestionLabel`, `AnswerJudgment` и `Span`.
4. Загружать `en_verb_inflections.txt`.
5. Строить прямой индекс `lemma -> paradigm`.
6. Строить обратный индекс `surface_form -> possible paradigms`.
7. Учитывать омонимию: одна surface form может соответствовать нескольким парадигмам.
8. Разбирать QA-SRL-вопрос на слоты.
9. Собирать канонический вопрос из слотов и формы глагола.
10. Проверять допустимость вопроса через Python-версию state machine.
11. Извлекать валидные answer spans и голоса разметчиков.
12. Сохранять совместимость с исходными именами полей QA-SRL.

Обязательный round-trip тест:

```python
parsed = parse_question(question_string, verb_inflected_forms)
rendered = render_question(parsed, verb_inflected_forms)
assert rendered == question_string
```

### Пока не переносить

- ScalaJS browser;
- Mechanical Turk pipeline;
- старый crowdsourcing UI;
- `qasrl-bank-service`;
- старый HTTP-сервер;
- все приложения из `apps/` без доказанной необходимости.

## Семантический генератор

Генератор строится на типизированных frames. Он не должен случайно заменять любое слово любым другим словом.

### Пример action frame `give`

```text
predicate_type = ACTION
agent = PERSON | ORGANIZATION
recipient = PERSON | ANIMAL | ORGANIZATION
theme = PHYSICAL_OBJECT | ABSTRACT_ENTITY
location = LOCATION, optional
time = DATE | TIME, optional
```

Поверхностные шаблоны:

```text
{agent} gave {recipient} {theme}.
{agent} gave {theme} to {recipient}.
{theme} was given to {recipient} by {agent}.
```

### Пример action frame `visit`

```text
predicate_type = ACTION
visitor = PERSON | ANIMAL
destination = LOCATION | ORGANIZATION
time = DATE | TIME, optional
```

### Пример state frame `own`

```text
predicate_type = STATE
owner = PERSON | ORGANIZATION
possession = PHYSICAL_OBJECT | ABSTRACT_ENTITY
```

### Пример state/emotion frame `feel`

```text
predicate_type = STATE
experiencer = PERSON | ANIMAL
emotion = EMOTION
cause = EVENT, optional
```

### Вариативность поверхности

Для одной семантической записи создавать варианты:

- времена;
- активный и пассивный залог;
- отрицание;
- модальность;
- вопросительный порядок слов;
- местоимения;
- придаточные предложения;
- фразовые глаголы;
- перестановка обстоятельств;
- свойства и степени свойств;
- нейтральная и разговорная речь.

Пример:

```text
Anna gave Rex a ball.
Anna gives Rex a ball.
Anna will give Rex a ball.
Anna did not give Rex a ball.
Did Anna give Rex a ball?
Rex was given a ball by Anna.
The ball that Anna gave Rex was red.
```

### Онтологическое наследование

Генератор должен уметь строить выводы:

```text
city -> LOCATION
dog -> ANIMAL
woman -> PERSON
car -> PHYSICAL_OBJECT
ship -> PHYSICAL_OBJECT
joy -> ABSTRACT_ENTITY + EMOTION
```

Пример:

```text
Paris is a city.
Every city is a location.
Therefore Paris is a location.
```

### Реальные и вымышленные имена

Использовать оба вида:

```text
Paris, London, Kyiv
Zelora, Narev, Taldin
```

Вымышленные имена нужны для проверки настоящего контекстного обобщения, а не запоминания мировых фактов.

### Не закреплять имя за одним классом

Не допускать, чтобы модель всегда учила:

```text
Rex -> ANIMAL
Paris -> LOCATION
Rose -> PERSON
```

Создавать контекстно неоднозначные случаи:

```text
Rex, a German shepherd, ran to the gate. -> ANIMAL
Rex, the new mechanic, fixed the car.    -> PERSON

They arrived in Paris before noon.       -> LOCATION
Paris called Helen after work.           -> PERSON

The jaguar chased its prey.              -> ANIMAL
The Jaguar stopped near the house.       -> PHYSICAL_OBJECT или NAMED_OBJECT
```

Также создавать случаи недостаточной информации:

```text
Anna spoke to Rex.
Question: Is Rex a person or an animal?
Answer: The text does not provide enough information.
```

## Семантический разметчик реальных текстов

### Вход

Поддержать минимум:

```text
*.txt
*.jsonl
```

Рекомендуемая запись входного JSONL:

```json
{"document_id":"book-00042","text":"Full document text...","source":"bookcorpus","license":"unknown"}
```

Это внутренний файловый формат, а не формат ответа обучаемой модели.

### Обработка

1. Сохранить исходный текст без необратимой нормализации.
2. Посчитать стабильный SHA-256.
3. Удалять точные дубли до аннотации.
4. Делить train/dev/test на уровне документов, а не отдельных предложений.
5. Разбивать документы на passages с сохранением глобальных char offsets.
6. Для обычного текста использовать контекст нескольких соседних предложений.
7. Для диалога сохранять предыдущие реплики, speaker roles и порядок ходов.
8. Создавать кандидатов через морфологию, POS/dependency parsing и QA-SRL.
9. Передавать кандидаты и контекст teacher-модели.
10. Валидировать, что каждый извлечённый span точно существует в исходном тексте.
11. Выполнять независимый verification pass.
12. Не перезаписывать старые аннотации: создавать новую `AnnotationRun`.

### Teacher adapter

Не привязывать ядро к одному провайдеру. Определить протокол адаптера, поддерживающий:

- локальную модель через HTTP/FastAPI;
- llama.cpp worker;
- OpenAI API;
- Bedrock или другой совместимый сервис.

Использование Hugging Face запрещено. Не использовать `transformers`, `datasets`, HF Hub или HF cache.

## Генератор вопросов — отдельная задача

Генератор вопросов принимает:

- исходный текст или диалог;
- проверенные spans;
- predicates;
- relations;
- dialogue annotations.

Он создаёт обычные вопросы и естественные ответы. Целевой ответ модели не должен быть JSON.

Пример:

```text
Text: On Monday, Anna gave her dog Rex a red ball in Kyiv.
Question: Who gave Rex a ball?
Answer: Anna.
```

Из одной семантической записи создавать несколько независимых примеров, сохраняя необходимый контекст в каждом:

```text
What did Anna do?
Who received the ball?
What was given?
What color was the ball?
Where did this happen?
When did this happen?
What kind of entity is Rex?
```

Дробить вопросы полезно. Нельзя дробить контекст, без которого ответ становится неоднозначным.

Для коротких реакций сохранять диалоговый контекст:

```text
A: I finally got the job.
B: Wow, that's great!

Question: What does “Wow” express?
Answer: Surprise and admiration.
```

### Типы QA-примеров

1. Один факт — один вопрос.
2. Один текст — составной вопрос.
3. Контекстные вопросы по нескольким предложениям или репликам.
4. Вопросы об онтологическом наследовании.
5. Вопросы о типе сущности.
6. Вопросы о свойствах, состояниях и эмоциях.
7. Вопросы о speech act, polarity и stance.
8. Вопросы без достаточной информации.
9. Yes/no-вопросы с объяснением.
10. Перефразированные варианты одного отношения.

### Негативные примеры

Добавлять корректный ответ:

```text
The text does not say.
The text does not provide enough information.
It is impossible to determine from the context.
```

Негативные примеры нужны, чтобы модель не выдумывала ответы.

### Формат обучения decoder-only модели

Рекомендуемая последовательность:

```text
<context>
On Monday, Anna gave her dog Rex a red ball in Kyiv.
</context>
<question>
Where did Anna give Rex the ball?
</question>
<answer>
In Kyiv.
</answer>
```

Loss должен считаться только по части `<answer>`, если training pipeline это поддерживает.

Хранить записи технически можно в JSONL, SQLite или Parquet. Это не означает, что модель должна генерировать JSON.

## Проверка обобщения

Создать отдельные тестовые поднаборы:

| Набор | Назначение |
|---|---|
| `standard_test` | Знакомые структуры и распределения |
| `unseen_names_test` | Новые имена при знакомых типах |
| `unseen_actions_test` | Новые глаголы или леммы |
| `unseen_combinations_test` | Знакомые элементы в новых сочетаниях |
| `ambiguous_names_test` | Выбор типа по контексту |
| `ontology_test` | Вывод `city -> LOCATION`, `dog -> ANIMAL` |
| `dialogue_act_test` | Значение `Oh`, `Wow`, `Hmm`, `Okay`, `No`, `Good` в контексте |
| `no_answer_test` | Способность не придумывать отсутствующую информацию |

Имена из `unseen_names_test` нельзя использовать в train. Генератор должен позволять выделять entity pools, action pools и combinations отдельно для каждого split.

## Хранилище и версионирование

Для внутреннего состояния использовать SQLite на первом этапе.

Рекомендуемые таблицы:

```text
documents
passages
utterances
annotation_runs
spans
relations
generation_runs
qa_examples
dataset_versions
```

Сохранять:

```text
source_document
source_hash
document_split
source_license
ontology_version
generator_version
model_name
prompt_version
random_seed
generation_time
confidence
review_status
synthetic
```

Требования:

- append-only аннотации;
- воспроизводимая генерация по seed;
- возможность пересобрать экспорт;
- невозможность смешать разные версии онтологии незаметно;
- дедупликация до разбиения и после генерации;
- отсутствие document leakage между train/dev/test.

## Предлагаемая структура Python-проекта

```text
semantic-corpus/
├── pyproject.toml
├── README.md
├── LICENSE
├── THIRD_PARTY_NOTICES.md
├── src/
│   └── semantic_corpus/
│       ├── qasrl_core/
│       │   ├── models.py
│       │   ├── bank_reader.py
│       │   ├── inflections.py
│       │   ├── question_slots.py
│       │   ├── question_parser.py
│       │   ├── question_renderer.py
│       │   ├── state_machine.py
│       │   └── validation.py
│       ├── ontology/
│       │   ├── models.py
│       │   ├── hierarchy.py
│       │   └── defaults.yaml
│       ├── semantic_generator/
│       │   ├── frames.py
│       │   ├── substitutions.py
│       │   ├── realization.py
│       │   └── transforms.py
│       ├── semantic_annotator/
│       │   ├── candidates.py
│       │   ├── teacher.py
│       │   ├── alignment.py
│       │   └── verifier.py
│       ├── question_generator/
│       │   ├── templates.py
│       │   ├── paraphrases.py
│       │   ├── answers.py
│       │   └── negatives.py
│       ├── storage/
│       │   ├── schema.py
│       │   └── repository.py
│       ├── exporters/
│       │   ├── sft.py
│       │   ├── jsonl.py
│       │   └── bio.py
│       └── cli.py
└── tests/
    ├── test_bank_reader.py
    ├── test_inflections.py
    ├── test_question_roundtrip.py
    ├── test_ontology.py
    ├── test_generation.py
    ├── test_alignment.py
    └── fixtures/
```

## Технические ограничения

- Python 3.10.
- Совместимость с Windows и WSL.
- Использовать `pathlib`, не строить пути вручную через `/`.
- Потоковое чтение больших JSONL/JSONL.GZ.
- Не загружать весь корпус в память.
- Не использовать Hugging Face.
- Базовые зависимости держать минимальными.
- Разрешены Pydantic, spaCy или Stanza, но тяжёлые компоненты должны быть опциональными extras.
- Для тестов использовать pytest.
- Отделить чистую доменную логику от API, CLI и базы данных.
- Все случайные операции обязаны принимать seed.

## Этапы реализации

### Этап 0. Аудит исходного репозитория

1. Клонировать `julianmichael/qasrl`.
2. Найти Scala-классы, соответствующие `Sentence`, `VerbEntry`, `QuestionLabel`, `AnswerJudgment`, `SlotBasedLabel`, `TemplateStateMachine`, `Frame`, `Tense`.
3. Составить таблицу `Scala source -> Python module`.
4. Не начинать массовый перенос до составления этой таблицы.

### Этап 1. Минимальное Python-ядро QA-SRL

1. Создать пакет и модели данных.
2. Реализовать потоковый reader QA-SRL Bank.
3. Реализовать загрузчик Wiktionary inflections и reverse lookup.
4. Реализовать question slots и renderer.
5. Добавить round-trip тесты на реальных QA-SRL записях.
6. Реализовать parser и начальную validation layer.

### Этап 2. Онтология и семантический генератор

1. Реализовать иерархию типов.
2. Реализовать typed slots.
3. Добавить frames `give`, `visit`, `own`, `feel`, `move`, `see`, `say`.
4. Добавить реальные и вымышленные name pools.
5. Добавить active/passive, tense и negation transformations.
6. Добавить генерацию неоднозначных случаев.
7. Добавить deterministic generation по seed.

### Этап 3. Семантический разметчик

1. Добавить ingestion TXT/JSONL.
2. Добавить разбиение документов с offsets.
3. Добавить candidate extraction.
4. Определить teacher adapter protocol.
5. Добавить alignment и invariant validation.
6. Добавить второй verification pass.

### Этап 4. Генератор вопросов

1. Генерировать atomic QA из relations.
2. Генерировать compound QA.
3. Генерировать entity type и ontology questions.
4. Генерировать dialogue-act questions.
5. Генерировать no-answer examples.
6. Добавить paraphrase layer.

### Этап 5. Хранилище и экспорты

1. SQLite schema и migrations.
2. Версионирование прогонов.
3. Документные splits.
4. Экспорт SFT.
5. Экспорт BIO/BILOU как производный формат.
6. Отчёт о распределении классов и утечках.

## Критерии готовности первого milestone

Первый milestone считается завершённым, когда:

1. Пакет устанавливается в Python 3.10.
2. QA-SRL `jsonl.gz` читается потоково.
3. Реальная запись корректно преобразуется в Python-модели.
4. `give/gave/given`, `go/went/gone` находятся через inflection index.
5. Обратный индекс возвращает несколько кандидатов при омонимии.
6. Question renderer воспроизводит исходный вопрос для тестового набора.
7. Некорректные spans и индексы отклоняются валидатором.
8. Есть pytest-тесты и они проходят.
9. Нет зависимости от Hugging Face.
10. README содержит короткий рабочий пример.

## Первая задача для Codex

Начать только с этапов 0 и 1. Не реализовывать сразу весь генератор.

Конкретная последовательность:

1. Проинспектировать структуру `julianmichael/qasrl` и перечислить переносимые Scala-классы.
2. Создать Python-проект по структуре выше, но сначала заполнить только `qasrl_core`.
3. Реализовать модели данных QA-SRL Bank.
4. Реализовать потоковый reader `.jsonl.gz`.
5. Реализовать `VerbInflectionParadigm` и индексы Wiktionary.
6. Реализовать базовый question renderer.
7. Добавить тесты.
8. Запустить тесты и показать результат.
9. Отдельно перечислить ещё не перенесённые правила state machine.

Не изменять исходный Scala-репозиторий. Создать отдельный Python-проект. При переносе алгоритмов сохранить MIT attribution.

## Принятые решения, которые не нужно повторно согласовывать

- `ACTION` и `STATE` разделены.
- `PROPERTY` добавлен.
- `EMOTION` может пересекаться с `STATE`, `PROPERTY` и `ABSTRACT_ENTITY`.
- `NAMED_ENTITY` является дополнительной меткой или атрибутом.
- Вложенные и multi-label spans разрешены.
- `DISCOURSE_MARKER`, `SPEECH_ACT`, `POLARITY`, `STANCE`, `MOOD` предусмотрены.
- Вопросный генератор и семантический генератор являются разными модулями.
- Ответ обучаемой модели не обязан быть JSON.
- Один текст можно дробить на много QA-примеров, но необходимый контекст сохраняется в каждом.
- Добавляются no-answer примеры.
- Используются реальные и вымышленные имена.
- Для проверки обобщения создаются held-out names, actions и combinations.
- Финальное решение о типе принимается по контексту, а не по словарю.
- Hugging Face не используется.

## Вопросы, которые можно отложить до следующих milestones

- Конкретный teacher provider.
- Финальный набор подтипов properties.
- Полный перечень action/state frames.
- UI для ручной проверки.
- Формат распределённой генерации.
- Поддержка языков, кроме английского.
- Постоянное запоминание новых фактов моделью: fine-tuning против внешнего knowledge store/RAG.

Эти вопросы не блокируют этапы 0 и 1.
