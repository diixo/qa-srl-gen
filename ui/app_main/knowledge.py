"""Architecture guide content; task membership comes from the running generator."""

from .chat_dataset import TASKS, TASK_LABELS, USER_ASSISTANT_TASKS


MODULES = [
    {
        'id': 'documents', 'name': 'Documents & ontology', 'layer': 'Core', 'icon': 'schema',
        'summary': 'Общий язык для текста, участников, предикатов, свойств и отношений.',
        'input': 'Текст и версии онтологии.',
        'output': 'Document, Passage, Utterance, TextSpan и AnnotationRun.',
        'detail': 'Символьные позиции относятся к исходному документу. Метки могут сочетаться: STATE + PROPERTY + EMOTION. Новый проход разметки создаёт отдельный AnnotationRun.',
        'files': ['semantic_corpus/documents.py', 'semantic_corpus/ontology/'],
    },
    {
        'id': 'pool', 'name': 'BuiltInPool', 'layer': 'Resources', 'icon': 'hub',
        'summary': 'Единая точка доступа к сущностям, ручным фреймам и намайненным ресурсам.',
        'input': 'EntityPool, DEFAULT_FRAMES, общий TemplateBank.',
        'output': 'Типизированные заполнения, фреймы, шаблоны и словарные инвентари.',
        'detail': 'Один логический пул объединяет разные виды ресурсов. Именная группа не становится типизированной сущностью автоматически. Контролируемые глаголы обрабатываются отдельной грамматикой.',
        'files': ['semantic_corpus/resource_pool.py', 'semantic_corpus/controlled_verbs.py', 'semantic_corpus/semantic_generator/entities.json'],
    },
    {
        'id': 'mining', 'name': 'Syntax mining & compiler', 'layer': 'Resources', 'icon': 'manage_search',
        'summary': 'Преобразование исходного текста в шаблоны, слоты и наблюдаемые формы.',
        'input': 'Текстовые записи и синтаксический разбор английского текста.',
        'output': 'CorpusTemplate, вхождения, глаголы, прилагательные и конструкции.',
        'detail': 'syntax_mining собирает наблюдения; template_compiler строит шаблоны; template_validation проверяет структуру и восстановление текста. Успешная структурная проверка сама по себе не подтверждает смысл.',
        'files': ['semantic_corpus/syntax_mining.py', 'semantic_corpus/template_compiler.py', 'semantic_corpus/template_validation.py'],
    },
    {
        'id': 'admission', 'name': 'Bank & admission', 'layer': 'Resources', 'icon': 'verified_user',
        'summary': 'Хранение, публикация и точные границы разрешённого использования.',
        'input': 'Проверенные архивы, кандидатные ресурсы и решения о конкретных вхождениях.',
        'output': 'Доступные компоненты банка и ограничения для генераторов.',
        'detail': 'manifest.json закрепляет опубликованные диапазоны; pending.json сохраняет кандидатные диапазоны и решения. template_usage ограничивает применение, reviewed_substitutions разрешает только явно проверенные варианты.',
        'files': ['semantic_corpus/template_bank.py', 'semantic_corpus/template_registry.py', 'semantic_corpus/template_pending.py', 'semantic_corpus/template_usage.py', 'semantic_corpus/reviewed_substitutions.py'],
    },
    {
        'id': 'synthetic', 'name': 'Semantic generator', 'layer': 'Generation', 'icon': 'auto_awesome',
        'summary': 'Создание предложений из семантических фреймов и совместимых заполнений.',
        'input': 'Фрейм, слоты, сущности, формы глаголов и параметры генерации.',
        'output': 'Текст, семантическая структура и точные привязки к фрагментам.',
        'detail': 'frames описывает роли и конструкции; substitutions выбирает допустимые сущности; realization и agreement согласуют формы. Built-in режим также использует намайненные шаблоны с их собственными ограничениями.',
        'files': ['semantic_corpus/semantic_generator/frames.py', 'semantic_corpus/semantic_generator/generator.py', 'semantic_corpus/semantic_generator/realization.py', 'semantic_corpus/agreement.py', 'semantic_corpus/template_semantics.py'],
    },
    {
        'id': 'annotation', 'name': 'Semantic annotator', 'layer': 'Annotation', 'icon': 'edit_note',
        'summary': 'Привязка предложенной семантической разметки к реальному тексту.',
        'input': 'Document, кандидаты и ответы выбранного teacher.',
        'output': 'AnnotationRun, принятые привязки и причины отклонений.',
        'detail': 'candidates предлагает фрагменты, alignment находит точные span, verifier проверяет структуру. Rule teacher покрывает ограниченную разметку диалоговых актов; HTTP teacher является отдельным опциональным режимом.',
        'files': ['semantic_corpus/semantic_annotator/pipeline.py', 'semantic_corpus/semantic_annotator/candidates.py', 'semantic_corpus/semantic_annotator/alignment.py', 'semantic_corpus/semantic_annotator/verifier.py'],
    },
    {
        'id': 'questions', 'name': 'QA & QA-SRL core', 'layer': 'Annotation', 'icon': 'question_answer',
        'summary': 'Вопросы и извлекаемые ответы по поддерживаемой семантической структуре.',
        'input': 'Предикаты, аргументы, свойства, отношения и исходные span.',
        'output': 'QA-примеры, варианты вопросов и результаты round-trip проверки.',
        'detail': 'question_generator строит QA по разметке. qasrl_core содержит грамматику вопросов, слоты, формы, parser/renderer и чтение QA-SRL Bank. Полнота синтаксического шаблона не гарантирует поддержку всех вопросов.',
        'files': ['semantic_corpus/question_generator/', 'semantic_corpus/qasrl_core/', 'semantic_corpus/qasrl_bridge.py'],
    },
    {
        'id': 'dialogues', 'name': 'Dialogue generator', 'layer': 'Generation', 'icon': 'forum',
        'summary': 'Контролируемые сценарии, в которых реплики и память следуют одному плану.',
        'input': 'Task, seed, профиль участников, квоты и ресурсы общего пула.',
        'output': 'Episode: реплики, изменения памяти, вопрос, ответ и свидетельства.',
        'detail': 'models хранит Fact, Memory, Delta, Transition, Query и Episode. actions и event_scenarios различают предложение, намерение и выполненное действие; fragments подключает проверенные интерпретации реплик.',
        'files': ['semantic_corpus/dialogue_generator/generator.py', 'semantic_corpus/dialogue_generator/models.py', 'semantic_corpus/dialogue_generator/actions.py', 'semantic_corpus/dialogue_generator/event_scenarios.py', 'semantic_corpus/dialogue_generator/fragments.py'],
    },
    {
        'id': 'mined-dialogues', 'name': 'Mined predicates → dialogues', 'layer': 'Generation', 'icon': 'account_tree',
        'summary': 'Адаптация намайненных предикатов к субъектам, аргументам и области действия.',
        'input': 'BuiltInPool, наблюдаемые клаузы, словарь форм Wiktionary и правила допуска.',
        'output': 'Поддерживаемые композиционные события либо цитаты с ограниченной интерпретацией.',
        'detail': 'mined_review проверяет пригодность; mined_subjects и mined_arguments связывают участников; mined_scope и verb_operators сохраняют модальность, отрицание и вложенность. Словарь форм не определяет семантику сам по себе.',
        'files': ['semantic_corpus/dialogue_generator/mined_bank.py', 'semantic_corpus/dialogue_generator/mined_review.py', 'semantic_corpus/dialogue_generator/mined_scope.py', 'semantic_corpus/dialogue_generator/mined_arguments.py', 'semantic_corpus/dialogue_generator/mined_subjects.py'],
    },
    {
        'id': 'export', 'name': 'Storage & export', 'layer': 'Output', 'icon': 'inventory_2',
        'summary': 'Сохранение корпусов, подготовка обучающих представлений и контроль целостности.',
        'input': 'Документы с разметкой и QA либо диалоговые Episode.',
        'output': 'JSONL corpus store, SFT/BIO/BILOU, History/Memory/Extraction и manifest.',
        'detail': 'Корпус хранится в JSONL-таблицах. Диалоговый export строит парные ответы History/Memory и отдельные примеры обновления памяти, проверяет квоты, дубликаты и разделение выборок.',
        'files': ['semantic_corpus/storage/', 'semantic_corpus/exporters/', 'semantic_corpus/dialogue_generator/export.py', 'semantic_corpus/dialogue_generator/memory_text.py', 'semantic_corpus/dialogue_generator/extraction_text.py'],
    },
    {
        'id': 'trainer', 'name': 'GPT-2 SFT trainer', 'layer': 'Training', 'icon': 'model_training',
        'summary': 'Отдельный CLI для обучения и оценки на подготовленных диалоговых данных.',
        'input': 'Выбранное представление датасета, GPT-2 и параметры обучения.',
        'output': 'Checkpoint, метрики и сравнение результатов.',
        'detail': 'Использует transformers Trainer и ChatML. SFT-loss относится к финальному ответу ассистента; превышение длины не исправляется молчаливым обрезанием. Обучение не подключено к UI.',
        'files': ['semantic_corpus/dialogue_generator/trainer/engine.py', 'semantic_corpus/dialogue_generator/trainer/chatml.py', 'semantic_corpus/dialogue_generator/trainer/metrics.py', 'semantic_corpus/dialogue_generator/trainer/__main__.py'],
    },
    {
        'id': 'ui', 'name': 'UI & execution', 'layer': 'Orchestration', 'icon': 'dashboard',
        'summary': 'Формы, постоянная очередь заданий, фоновые процессы и просмотр результатов.',
        'input': 'Настройки операции через Django UI или команды CLI.',
        'output': 'Job, статус, журнал и файлы результата.',
        'detail': 'operations.json задаёт каталог операций; worker исполняет зарегистрированные обработчики. SQLite хранит историю заданий и настройки UI. Сам corpus store и Built-in bank находятся в файлах.',
        'files': ['ui/app_main/views.py', 'ui/app_main/operations.json', 'ui/app_main/operations.py', 'ui/app_main/worker.py', 'semantic_corpus/cli.py'],
    },
]

TASK_NOTES = {
    'recall': ('Факты и память', 'Вспомнить ранее сообщённый факт.'),
    'correction': ('Факты и память', 'Заменить устаревшее значение после исправления.'),
    'different_people': ('Факты и память', 'Не смешивать факты о разных людях.'),
    'negation': ('Факты и память', 'Различать отрицание и отсутствие информации.'),
    'unknown': ('Факты и память', 'Отвечать об отсутствии знания, не выдумывая значение.'),
    'topic_return': ('Факты и память', 'Вернуться к факту после смены темы.'),
    'retraction': ('Факты и память', 'Отозвать факт, не превращая отзыв в противоположный факт.'),
    'desire_vs_done': ('Намерения и действия', 'Отделять желание или план от состоявшегося действия.'),
    'suggestion_vs_done': ('Намерения и действия', 'Не считать предложение ассистента выполненным действием.'),
    'capability': ('Намерения и действия', 'Сохранять границу между возможностями ассистента и реальным выполнением.'),
    'option_reference': ('Контекст и ссылки', 'Разрешать «первый / второй» по порядку предложенных вариантов.'),
    'ambiguous_reference': ('Контекст и ссылки', 'Запрашивать уточнение, если ссылка неоднозначна.'),
    'event_scope': ('События', 'Учитывать вложенные намерения, необходимость, модальность и фразовые действия.'),
    'event_progress': ('События', 'Различать попытку, процесс и завершение.'),
    'event_updates': ('События', 'Исправлять и отзывать события, сохраняя остальные действия.'),
    'event_states': ('События', 'Различать местонахождение, владение, желания и прошлые состояния.'),
    'mined_predicates': ('Намайненные ресурсы', 'Использовать предикаты через проверяемые конструкции или ограниченные цитаты.'),
    'fragment_meaning': ('Намайненные ресурсы', 'Извлекать доступный смысл коротких, полных и незаконченных реплик с контекстом.'),
}


def knowledge_context():
    groups = {}
    for code in TASKS:
        group, description = TASK_NOTES.get(code, ('Другие задачи', 'Описание этой задачи ещё не добавлено в Knowledge.'))
        groups.setdefault(group, []).append({
            'code': code, 'label': TASK_LABELS.get(code, code), 'description': description,
            'named_only': code not in USER_ASSISTANT_TASKS,
        })
    return {'modules': MODULES, 'task_groups': groups.items(),
            'task_count': len(TASKS), 'simple_task_count': len(USER_ASSISTANT_TASKS)}
