(() => {
  'use strict';
  const root = document.querySelector('.knowledge');
  if (!root) return;

  const search = root.querySelector('#knowledge-query');
  const clear = root.querySelector('[data-search-clear]');
  const count = root.querySelector('#knowledge-search-count');
  const items = [...root.querySelectorAll('[data-search-item]')];
  const groups = [...root.querySelectorAll('[data-task-group]')];
  const normalize = (value) => value.normalize('NFKC').toLocaleLowerCase().trim();
  const searchable = new Map(items.map((item) => [item, normalize(item.textContent)]));

  function filter() {
    const words = normalize(search.value).split(/\s+/).filter(Boolean);
    for (const item of items) {
      item.hidden = !words.every((word) => searchable.get(item).includes(word));
    }
    for (const group of groups) {
      group.hidden = [...group.querySelectorAll('[data-search-item]')].every((item) => item.hidden);
    }
    const modules = items.filter((item) => item.matches('.knowledge-module') && !item.hidden).length;
    const tasks = items.filter((item) => item.matches('.knowledge-task') && !item.hidden).length;
    root.querySelector('[data-empty-modules]').hidden = modules > 0;
    root.querySelector('[data-empty-tasks]').hidden = tasks > 0;
    clear.hidden = !search.value;
    count.textContent = words.length ? `Найдено модулей: ${modules}; задач: ${tasks}.` : '';
  }
  root.querySelector('[data-knowledge-search]').hidden = false;
  search.addEventListener('input', filter);
  clear.addEventListener('click', () => { search.value = ''; filter(); search.focus(); });
  search.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') { search.value = ''; filter(); }
  });

  function revealAnchor() {
    const target = document.getElementById(window.location.hash.slice(1));
    if (!target || !root.contains(target)) return;
    if (target.matches('.knowledge-module')) {
      if (target.hidden) { search.value = ''; filter(); }
      target.open = true;
    }
    target.scrollIntoView({block: 'start', behavior: 'instant'});
    if (target === search) search.focus({preventScroll: true});
  }
  window.addEventListener('hashchange', revealAnchor);
  root.addEventListener('click', (event) => {
    const anchor = event.target.closest('a[href^="#"]');
    if (anchor && anchor.getAttribute('href') === window.location.hash) revealAnchor();
  });
  revealAnchor();

  const sections = [...root.querySelectorAll('.knowledge-section')];
  const links = [...root.querySelectorAll('.knowledge-toc nav a')];
  let queued = false;
  function trackSection() {
    queued = false;
    const current = sections.filter((section) => section.getBoundingClientRect().top <= 150).pop() || sections[0];
    for (const link of links) {
      if (link.hash === `#${current.id}`) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    }
  }
  window.addEventListener('scroll', () => {
    if (!queued) { queued = true; window.requestAnimationFrame(trackSection); }
  }, {passive: true});
  trackSection();
})();
