/**
 * API client.
 *
 * Every call goes through `request()`, so auth, error shaping and JSON parsing
 * exist in exactly one place. The server returns `{error, detail}` on failure;
 * `detail` is written to be shown to a person, so that is what surfaces.
 */

const TOKEN_KEY = 'scanner.token';

export class ApiError extends Error {
  constructor(message, status, code) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
  }
}

/** Auth token, remembered per browser so a reload does not re-prompt. */
export const token = {
  get() {
    try {
      return localStorage.getItem(TOKEN_KEY) || '';
    } catch {
      return ''; // private mode / blocked storage
    }
  },
  set(value) {
    try {
      value ? localStorage.setItem(TOKEN_KEY, value) : localStorage.removeItem(TOKEN_KEY);
    } catch {
      /* non-fatal: the token just will not persist */
    }
  },
};

async function request(path, { method = 'GET', body, signal } = {}) {
  const headers = {};
  const t = token.get();
  if (t) headers.Authorization = `Bearer ${t}`;

  let response;
  try {
    response = await fetch(path, { method, body, headers, signal, cache: 'no-store' });
  } catch (err) {
    if (err.name === 'AbortError') throw err;
    throw new ApiError('Cannot reach the server. Is the service running?', 0, 'network');
  }

  if (response.status === 204) return null;

  const isJson = (response.headers.get('content-type') || '').includes('json');
  const payload = isJson ? await response.json().catch(() => null) : null;

  if (!response.ok) {
    throw new ApiError(
      payload?.detail || `Request failed with status ${response.status}`,
      response.status,
      payload?.error || 'error',
    );
  }
  return payload;
}

/**
 * Append the token to a URL used by <img>, <a download> or a 3D loader —
 * those cannot send an Authorization header.
 */
export function withToken(url) {
  const t = token.get();
  if (!t) return url;
  const joiner = url.includes('?') ? '&' : '?';
  return `${url}${joiner}token=${encodeURIComponent(t)}`;
}

export const api = {
  config: () => request('/api/config'),
  health: () => request('/api/health'),

  listScans: () => request('/api/scans'),
  getScan: (id, signal) => request(`/api/scans/${id}`, { signal }),
  deleteScan: (id) => request(`/api/scans/${id}`, { method: 'DELETE' }),
  reprocessScan: (id) => request(`/api/scans/${id}/reprocess`, { method: 'POST' }),

  createSample: (name) => {
    const form = new FormData();
    if (name) form.set('name', name);
    return request('/api/scans/sample', { method: 'POST', body: form });
  },

  /**
   * Upload a capture bundle. Uses XMLHttpRequest rather than fetch because it
   * is still the only way to report upload progress.
   */
  createUpload({ cloud, poses, frames = [], name = '', onProgress }) {
    const form = new FormData();
    form.set('cloud', cloud, cloud.name);
    if (poses) form.set('poses', poses, poses.name);
    for (const frame of frames) form.append('frames', frame, frame.name);
    if (name) form.set('name', name);

    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', '/api/scans');
      const t = token.get();
      if (t) xhr.setRequestHeader('Authorization', `Bearer ${t}`);

      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable && onProgress) onProgress(event.loaded / event.total);
      });
      xhr.addEventListener('load', () => {
        let payload = null;
        try {
          payload = JSON.parse(xhr.responseText);
        } catch {
          /* non-JSON error body */
        }
        if (xhr.status >= 200 && xhr.status < 300) resolve(payload);
        else {
          reject(new ApiError(
            payload?.detail || `Upload failed with status ${xhr.status}`,
            xhr.status,
            payload?.error || 'error',
          ));
        }
      });
      xhr.addEventListener('error', () => reject(new ApiError('Upload failed: network error.', 0, 'network')));
      xhr.addEventListener('abort', () => reject(new ApiError('Upload cancelled.', 0, 'aborted')));
      xhr.send(form);
    });
  },

  floorplan: (id, signal) => request(`/api/scans/${id}/floorplan`, { signal }),
  colliders: (id, signal) => request(`/api/scans/${id}/colliders`, { signal }),

  artifactUrl: (id, name) => withToken(`/api/scans/${id}/artifacts/${name}`),
  frameUrl: (id, file) => withToken(`/api/scans/${id}/frames/${String(file).split('/').pop()}`),
};
