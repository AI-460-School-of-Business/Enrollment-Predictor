document.getElementById("apiBtn").addEventListener("click", async () => {
  try {
    const res = await fetch("/api/hello");
    if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
    const data = await res.json();
    document.getElementById("apiResponse").innerText = data.message;
  } catch (err) {
    console.error("API call failed:", err);
    document.getElementById("apiResponse").innerText = "API call failed";
  }
});
