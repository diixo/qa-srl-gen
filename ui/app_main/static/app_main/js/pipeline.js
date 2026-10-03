/* Local UI: bounded polling and operation-specific controls. */
(() => {
  const form = document.querySelector('[data-operation-form]');
  const fields = {
    build: ['count', 'seed', 'split', 'pool', 'scheme', 'max_per_kind', 'no_answer_share'],
    dailydialog: ['source', 'limit', 'seed', 'context_turns'],
    ingest: ['source', 'input_format', 'limit', 'split', 'context_turns', 'seed'],
    bank: ['source', 'split', 'limit', 'scheme', 'seed'],
    annotate: ['store', 'teacher', 'teacher_url', 'model', 'limit', 'seed', 'scheme'],
    questions: ['store', 'limit', 'seed', 'scheme', 'max_per_kind', 'no_answer_share'],
    export: ['store', 'scheme'], report: ['store'],
    validate: ['source', 'limit'], roundtrip: ['source', 'limit'],
    candidates: ['text'], lookup: ['text'], mining: ['source', 'limit'],
  };
  const hints = {
    build: 'Генерация → сохранение аннотаций → QA → SFT / BIO → отчёт качества.',
    dailydialog: 'Офлайн-проход RuleBasedTeacher. Размечает диалоговые акты; типы сущностей и смыслы предикатов требуют другого учителя. Используется исходный train split.',
    ingest: 'TXT / JSONL: split назначается детерминированно по документу. Для диалогов используется выбранный split.',
    annotate: 'Правила размечают диалоговые акты. HTTP teacher получает текст выбранного корпуса по указанному URL.',
    mining: 'Предложения сохраняются с подтверждающими примерами. Перед добавлением в пул их нужно проверить.',
  };
  if (form) {
    const operation = form.querySelector('[name=operation]');
    const update = () => {
      const visible = new Set(['operation', ...(fields[operation.value] || [])]);
      form.querySelectorAll('[data-field]').forEach(el => {
        el.hidden = !visible.has(el.dataset.field);
        el.querySelectorAll('input, select, textarea').forEach(input => { input.disabled = el.hidden; });
      });
      form.querySelector('[data-operation-hint]').textContent = hints[operation.value] || 'Запуск выполнится в фоне. Результаты и лог будут доступны в истории.';
    };
    operation.addEventListener('change', update);
    update();
    form.addEventListener('submit', () => { form.querySelector('button[type=submit]').disabled = true; });
  }
  document.querySelectorAll('[data-refresh]').forEach(el => el.addEventListener('click', () => location.reload()));
  if (document.querySelector('[data-auto-refresh]')) {
    setInterval(() => {
      if (!document.hidden && !document.activeElement.closest('form')) location.reload();
    }, 10000);
  }
  const panel = document.querySelector('[data-job-status]');
  if (panel) {
    const poll = async () => {
      if (document.hidden) { setTimeout(poll, 2500); return; }
      try {
        const response = await fetch(panel.dataset.jobStatus, {cache: 'no-store'});
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const job = await response.json();
        if (!job.active) { location.reload(); return; }
        const status = panel.querySelector('[data-status]');
        status.textContent = job.cancel_requested ? 'Отмена запрошена' : job.label;
        status.className = `status-badge ${job.status}`;
        panel.querySelector('[data-phase]').textContent = job.phase;
        panel.querySelector('[data-elapsed]').textContent = job.elapsed_seconds;
        panel.querySelector('[data-count]').textContent = job.total ? `${job.processed} / ${job.total}` : `Обработано: ${job.processed}`;
        const bar = panel.querySelector('[data-progress]');
        bar.style.width = `${job.total ? job.percent : 100}%`;
        bar.classList.toggle('progress-bar-striped', !job.total);
        bar.classList.toggle('progress-bar-animated', !job.total);
        bar.parentElement.setAttribute('aria-valuenow', job.percent);
        panel.querySelector('[data-metrics]').textContent = Object.entries(job.metrics).map(([key, value]) => `${key}: ${value}`).join(' · ');
        panel.querySelector('[data-connection]').textContent = job.worker_online ? '' : 'Нет сигнала от воркера. Проверьте страницу «Воркеры и очередь».';
        document.querySelector('[data-log]').textContent = job.log || 'Ожидание вывода…';
      } catch (error) {
        panel.querySelector('[data-connection]').textContent = 'Не удалось обновить состояние. Повтор через несколько секунд.';
      }
      setTimeout(poll, 2500);
    };
    setTimeout(poll, 1500);
  }
})();
