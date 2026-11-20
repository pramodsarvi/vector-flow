/**
 * Small DOM helpers and the app's feedback surfaces (toasts, status line).
 *
 * Deliberately not a framework: the viewer has a handful of views, and a
 * create-element helper plus explicit updates is easier to follow — and to
 * keep accessible — than a reactive layer would be at this size.
 */

/** Create an element: el('button.btn.primary', {onclick}, 'Label') */
export function el(spec, attrs = {}, ...children) {
  const [tag, ...classes] = spec.split('.');
  const node = document.createElement(tag || 'div');
  if (classes.length) node.className = classes.join(' ');

  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key.startsWith('on') && typeof value === 'function') {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (key === 'dataset') {
      Object.assign(node.dataset, value);
    } else if (key === 'html') {
      node.innerHTML = value; // only ever used with strings this app built
    } else if (key in node && key !== 'list' && key !== 'form') {
      node[key] = value;
    } else {
      node.setAttribute(key, value === true ? '' : value);
    }
  }
  append(node, children);
  return node;
}

export function append(parent, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    parent.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return parent;
}

export function clear(node) {
  while (node.firstChild) node.firstChild.remove();
  return node;
}

export const $ = (selector, root = document) => root.querySelector(selector);

/* ---------------------------------------------------------------- toasts */

let toastHost = null;

function host() {
  if (!toastHost) {
    // aria-live so a screen reader announces failures, which are otherwise
    // purely visual and easy to miss.
    toastHost = el('div.toasts', { id: 'toasts', role: 'status', 'aria-live': 'polite' });
    document.body.append(toastHost);
  }
  return toastHost;
}

export function toast(message, kind = 'info', timeout = 5000) {
  const node = el(`div.toast.${kind}`, {}, message);
  host().append(node);
  const remove = () => {
    node.classList.add('leaving');
    setTimeout(() => node.remove(), 200);
  };
  if (timeout) setTimeout(remove, timeout);
  node.addEventListener('click', remove);
  return remove;
}

/* ------------------------------------------------------- status & errors */

/** Panel-level error with a retry affordance, instead of a silently blank panel. */
export function errorPanel(message, onRetry) {
  return el('div.panel-state.error', { role: 'alert' },
    el('div.panel-state-icon', {}, '!'),
    el('p.panel-state-msg', {}, message),
    onRetry ? el('button.btn.small', { type: 'button', onclick: onRetry }, 'Try again') : null,
  );
}

export function emptyPanel(message, action) {
  return el('div.panel-state', {},
    el('p.panel-state-msg', {}, message),
    action || null,
  );
}

export function spinner(label = 'Loading') {
  return el('div.panel-state', { role: 'status' },
    el('div.spinner', { 'aria-hidden': 'true' }),
    el('p.panel-state-msg', {}, label),
  );
}

/* ------------------------------------------------------------ formatting */

export const fmt = {
  metres: (value) => (Number.isFinite(value) ? `${value.toFixed(2)} m` : '—'),
  area: (sqm, sqft) =>
    Number.isFinite(sqm) ? `${sqm.toFixed(2)} m²${Number.isFinite(sqft) ? ` · ${sqft.toFixed(0)} ft²` : ''}` : '—',
  count: (n, singular, plural = `${singular}s`) => `${n} ${n === 1 ? singular : plural}`,
  points: (n) => (Number.isFinite(n) ? n.toLocaleString() : '—'),
  duration: (s) => (Number.isFinite(s) ? `${s.toFixed(1)}s` : '—'),
  when(iso) {
    if (!iso) return '—';
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return '—';
    const mins = Math.round((Date.now() - date.getTime()) / 60000);
    if (mins < 1) return 'just now';
    if (mins < 60) return `${mins} min ago`;
    if (mins < 60 * 24) return `${Math.round(mins / 60)} h ago`;
    return date.toLocaleDateString();
  },
  bytes(n) {
    if (!Number.isFinite(n)) return '—';
    const units = ['B', 'KB', 'MB', 'GB'];
    let value = n;
    let unit = 0;
    while (value >= 1024 && unit < units.length - 1) {
      value /= 1024;
      unit += 1;
    }
    return `${value.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
  },
};
