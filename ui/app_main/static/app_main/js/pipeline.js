/* Local UI: bounded polling and operation-specific controls. */
(() => {
  const form = document.querySelector('[data-operation-form]');
  const fields = {
    build: ['count', 'template_count', 'recombine', 'seed', 'split', 'pool', 'scheme', 'max_per_kind', 'no_answer_share'],
    templates: ['template_count', 'recombine', 'seed', 'split'],
    dailydialog: ['source', 'limit', 'seed', 'context_turns'],
    ingest: ['source', 'input_format', 'limit', 'split', 'context_turns', 'seed'],
    bank: ['source', 'split', 'limit', 'scheme', 'seed'],
    annotate: ['store', 'pending_only', 'teacher', 'teacher_url', 'model', 'limit', 'seed', 'scheme'],
    questions: ['store', 'limit', 'seed', 'scheme', 'max_per_kind', 'no_answer_share'],
    export: ['store', 'scheme'], report: ['store'],
    validate: ['source', 'limit'], roundtrip: ['source', 'limit'],
    candidates: ['text'], lookup: ['text'], mining: ['source', 'limit'],
  };
  const hints = {
    templates: 'Render every matching corpus template into JSONL and a document store. Automatic syntax stays unreviewed. Use Corpus annotation next to obtain semantic labels and QA.',
    build: 'Built-in pool supplies curated frames and mined templates. Curated situations produce verified QA and SFT / BIO exports; mined documents retain unreviewed syntax and need semantic annotation before training export.',
    dailydialog: 'Offline RuleBasedTeacher pass for dialogue acts. Entity types and predicate senses need a different teacher. Uses the original train split.',
    ingest: 'TXT / JSONL: splits are assigned deterministically per document. Dialogues use the selected split.',
    annotate: 'Only documents awaiting annotation preserves existing annotations; the limit applies after this selection. Rules annotate dialogue acts. Use an HTTP teacher for semantic labels and roles.',
    mining: 'Proposals are saved with supporting examples. Review them before adding them to the pool.',
  };
  if (form) {
    const operation = form.querySelector('[name=operation]');
    const count = form.querySelector('[name=count]');
    const templateCount = form.querySelector('[name=template_count]');
    const update = () => {
      const visible = new Set(['operation', ...(fields[operation.value] || [])]);
      form.querySelectorAll('[data-field]').forEach(el => {
        el.hidden = !visible.has(el.dataset.field);
        el.querySelectorAll('input, select, textarea').forEach(input => { input.disabled = el.hidden; });
      });
      let hint = hints[operation.value] || 'The job runs in the background. Find its results and log in Job history.';
      if (operation.value === 'build' && count.value && count.validity.valid) {
        const frames = Number(count.dataset.frameCount);
        const curated = `${count.value} per frame × ${frames} curated frames = ${(Number(count.value) * frames).toLocaleString('en-US')} situations.`;
        const perTemplate = Number(templateCount.value === '' ? 1 : templateCount.value);
        let mined;
        if (perTemplate === 0) {
          mined = 'Mined templates are explicitly skipped.';
        } else {
          const templates = Number(templateCount.dataset.templateCount);
          mined = `${perTemplate.toLocaleString('en-US')} per mined template × ${templates.toLocaleString('en-US')} templates = ${(perTemplate * templates).toLocaleString('en-US')} candidates before split filtering and deduplication.`;
        }
        hint = `${curated} ${mined} ${hint}`;
      }
      form.querySelector('[data-operation-hint]').textContent = hint;
    };
    operation.addEventListener('change', update);
    [count, templateCount].forEach(input => input.addEventListener('input', update));
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
        status.textContent = job.cancel_requested ? 'Cancellation requested' : job.label;
        status.className = `status-badge ${job.status}`;
        panel.querySelector('[data-phase]').textContent = job.phase;
        panel.querySelector('[data-elapsed]').textContent = job.elapsed_seconds;
        panel.querySelector('[data-count]').textContent = job.total ? `${job.processed} / ${job.total}` : `Processed: ${job.processed}`;
        const bar = panel.querySelector('[data-progress]');
        bar.style.width = `${job.total ? job.percent : 100}%`;
        bar.classList.toggle('progress-bar-striped', !job.total);
        bar.classList.toggle('progress-bar-animated', !job.total);
        bar.parentElement.setAttribute('aria-valuenow', job.percent);
        panel.querySelector('[data-metrics]').textContent = Object.entries(job.metrics).map(([key, value]) => `${key}: ${value}`).join(' · ');
        panel.querySelector('[data-connection]').textContent = job.worker_online ? '' : 'No worker heartbeat. Check the Workers and queue page.';
        document.querySelector('[data-log]').textContent = job.log || 'Waiting for output…';
      } catch (error) {
        panel.querySelector('[data-connection]').textContent = 'Could not refresh the status. Retrying in a few seconds.';
      }
      setTimeout(poll, 2500);
    };
    setTimeout(poll, 1500);
  }
})();
