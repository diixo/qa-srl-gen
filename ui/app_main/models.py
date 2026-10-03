import uuid

from django.db import models
from django.utils import timezone


OPERATIONS = [
    ('build', 'Синтетический корпус'), ('dailydialog', 'DailyDialog: полный цикл'),
    ('ingest', 'Импорт текстов / диалогов'), ('bank', 'Импорт QA-SRL Bank'),
    ('annotate', 'Аннотация корпуса'), ('questions', 'Генерация QA'),
    ('export', 'Экспорт SFT / BIO'), ('report', 'Отчёт и проверка корпуса'),
    ('validate', 'Валидация QA-SRL Bank'), ('roundtrip', 'Round-trip вопросов'),
    ('candidates', 'Поиск кандидатов'), ('lookup', 'Глагольные формы'),
    ('mining', 'Кандидаты в пул сущностей'),
]


class Job(models.Model):
    class Status(models.TextChoices):
        QUEUED = 'queued', 'В очереди'
        RUNNING = 'running', 'Выполняется'
        SUCCEEDED = 'succeeded', 'Завершён'
        FAILED = 'failed', 'Ошибка'
        CANCELLED = 'cancelled', 'Отменён'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    operation = models.CharField(max_length=24, choices=OPERATIONS)
    config = models.JSONField(default=dict)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.QUEUED)
    phase = models.CharField(max_length=200, default='Ожидание свободного слота')
    processed = models.PositiveIntegerField(default=0)
    total = models.PositiveIntegerField(default=0)
    metrics = models.JSONField(default=dict)
    cancel_requested = models.BooleanField(default=False)
    created_at = models.DateTimeField(default=timezone.now)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    pid = models.PositiveIntegerField(null=True, blank=True)
    slot = models.PositiveIntegerField(null=True, blank=True)
    error = models.TextField(blank=True)

    class Meta:
        ordering = ['-created_at']

    @property
    def active(self):
        return self.status in (self.Status.QUEUED, self.Status.RUNNING)

    @property
    def percent(self):
        return min(100, round(100 * self.processed / self.total)) if self.total else 0

    @property
    def elapsed_seconds(self):
        if not self.started_at:
            return 0
        return round(((self.finished_at or timezone.now()) - self.started_at).total_seconds(), 1)


class Worker(models.Model):
    """One local supervisor, with a configurable number of isolated child slots."""

    id = models.PositiveSmallIntegerField(primary_key=True, default=1, editable=False)
    capacity = models.PositiveSmallIntegerField(default=2)
    pid = models.PositiveIntegerField(null=True, blank=True)
    heartbeat = models.DateTimeField(null=True, blank=True)
    stop_requested = models.BooleanField(default=False)

    @property
    def online(self):
        return bool(self.pid and self.heartbeat and
                    (timezone.now() - self.heartbeat).total_seconds() < 15)
