const scanBtn = document.getElementById('scanBtn')
const apiUrlInput = document.getElementById('apiUrl')
const healthBadge = document.getElementById('healthBadge')
const statusEl = document.getElementById('status')
const resultCard = document.getElementById('resultCard')
const resultLabel = document.getElementById('resultLabel')
const resultConfidence = document.getElementById('resultConfidence')
const meterFill = document.getElementById('meterFill')
const resultMeta = document.getElementById('resultMeta')
const resultRec = document.getElementById('resultRec')
const resultDisclaimer = document.getElementById('resultDisclaimer')

function getBaseUrl() {
  return apiUrlInput.value.trim().replace(/\/+$/, '') || 'http://localhost:8000'
}

async function checkBackendHealth() {
  healthBadge.textContent = 'Checking...'
  healthBadge.className = 'health-badge checking'
  try {
    const res = await fetch(`${getBaseUrl()}/health`)
    if (!res.ok) throw new Error('Unhealthy')
    const data = await res.json()
    healthBadge.textContent = data.model_loaded ? 'API Online' : 'No Model'
    healthBadge.className = `health-badge ${data.model_loaded ? 'online' : 'offline'}`
  } catch {
    healthBadge.textContent = 'API Offline'
    healthBadge.className = 'health-badge offline'
  }
}

chrome.storage?.local.get(['apiUrl'], (items) => {
  if (items?.apiUrl) {
    apiUrlInput.value = items.apiUrl
  }
  checkBackendHealth()
})

apiUrlInput.addEventListener('change', () => {
  chrome.storage?.local.set({ apiUrl: apiUrlInput.value.trim() })
  checkBackendHealth()
})

function extractArticleText() {
  const articleEl = document.querySelector('article') || document.querySelector('main')
  const paragraphs = Array.from((articleEl || document).querySelectorAll('p'))
    .map((p) => p.innerText.trim())
    .filter((t) => t.length > 30)

  if (paragraphs.length > 0) {
    return paragraphs.join('\n\n').slice(0, 12000)
  }
  return (document.body?.innerText || '').trim().slice(0, 12000)
}

scanBtn.addEventListener('click', async () => {
  const baseUrl = getBaseUrl()
  scanBtn.disabled = true
  resultCard.classList.add('hidden')
  statusEl.textContent = 'Extracting article text from current tab...'
  statusEl.classList.remove('hidden')

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true })
    if (!tab?.id) {
      throw new Error('No active browser tab found.')
    }

    const [{ result: extractedText }] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: extractArticleText,
    })

    if (!extractedText || extractedText.trim().length < 20) {
      throw new Error('Could not extract enough article text from this page.')
    }

    statusEl.textContent = 'Analyzing article...'

    const response = await fetch(`${baseUrl}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: extractedText,
        url: tab.url || null,
        source: 'chrome-extension',
      }),
    })

    const data = await response.json()
    if (!response.ok) {
      throw new Error(data.detail || 'Backend returned an error.')
    }

    const pct = Math.round(data.confidence * 100)
    statusEl.classList.add('hidden')
    resultCard.className = `result-card ${data.risk_level}`
    resultLabel.textContent = data.label
    resultConfidence.textContent = `${pct}%`
    meterFill.style.width = `${pct}%`
    resultMeta.textContent = `Language: ${data.language_name} · Matched features: ${data.matched_features}`
    resultRec.textContent = data.recommendation
    resultDisclaimer.textContent = data.disclaimer
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`
  } finally {
    scanBtn.disabled = false
  }
})
