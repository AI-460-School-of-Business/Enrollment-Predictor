document.getElementById('apiBtn').addEventListener('click', async () => {
  const el = document.getElementById('apiResponse');
  el.textContent = 'Testing connection...';
  try {
    const res = await fetch('/api/hello');
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    el.textContent = `✅ ${data.message}`;
  } catch (e) {
    el.textContent = `❌ ${e.message}`;
  }
});
