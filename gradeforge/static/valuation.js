/**
 * Card Valuation UI Module
 * Displays market valuation data after card grading
 */

// CSS Styles for Valuation Card
const valuationStyles = `
  .valuation-card {
    border:2px solid var(--accent);
    border-radius:12px;
    padding:20px;
    background:linear-gradient(135deg, rgba(59,130,246,0.08) 0%, transparent 100%);
    margin-top:20px;
  }
  .valuation-header {
    display:flex;
    align-items:center;
    gap:12px;
    margin-bottom:16px;
  }
  .valuation-icon {
    font-size:28px;
  }
  .valuation-title {
    font-size:18px;
    font-weight:700;
    color:var(--accent);
    margin:0;
  }
  .valuation-price {
    font-size:32px;
    font-weight:900;
    color:var(--accent);
    margin:12px 0;
  }
  .valuation-grid {
    display:grid;
    gap:12px;
    grid-template-columns:repeat(auto-fit, minmax(140px, 1fr));
  }
  .valuation-item {
    background:rgba(59,130,246,0.1);
    border:1px solid rgba(59,130,246,0.3);
    border-radius:8px;
    padding:12px;
  }
  .valuation-label {
    font-size:12px;
    color:var(--muted);
    font-weight:600;
    text-transform:uppercase;
    letter-spacing:0.5px;
  }
  .valuation-value {
    font-size:16px;
    font-weight:700;
    color:var(--text);
    margin-top:4px;
  }
  .valuation-confidence {
    display:inline-block;
    padding:4px 10px;
    border-radius:6px;
    font-size:12px;
    font-weight:600;
    margin-top:8px;
  }
  .valuation-confidence.high {
    background:rgba(16,185,129,0.2);
    color:var(--ok);
  }
  .valuation-confidence.medium {
    background:rgba(245,158,11,0.2);
    color:var(--warn);
  }
  .valuation-confidence.low {
    background:rgba(239,68,68,0.2);
    color:var(--bad);
  }
  .valuation-sources {
    font-size:12px;
    color:var(--muted);
    margin-top:12px;
  }
  .valuation-loading {
    font-size:14px;
    color:var(--muted);
    font-style:italic;
  }
  .valuation-error {
    color:var(--bad);
    font-size:14px;
  }
`;

// Inject styles into document
function injectValuationStyles() {
  const style = document.createElement('style');
  style.textContent = valuationStyles;
  document.head.appendChild(style);
}

// Fetch and display valuation for a graded card
async function fetchAndDisplayValuation(analysisId) {
  const resultDiv = document.getElementById('result');
  if (!resultDiv) return;

  // Create valuation container
  const valuationDiv = document.createElement('div');
  valuationDiv.className = 'valuation-card';
  valuationDiv.id = `valuation-${analysisId}`;

  // Show loading state
  valuationDiv.innerHTML = `
    <div class="valuation-header">
      <span class="valuation-icon">💰</span>
      <h3 class="valuation-title">Market Valuation</h3>
    </div>
    <div class="valuation-loading">Fetching market data...</div>
  `;

  // Add to results
  resultDiv.appendChild(valuationDiv);

  try {
    const response = await fetch(`/api/valuation/${analysisId}`);
    if (!response.ok) throw new Error('Valuation service unavailable');

    const data = await response.json();

    // Build valuation display
    let html = `
      <div class="valuation-header">
        <span class="valuation-icon">💰</span>
        <h3 class="valuation-title">Market Valuation</h3>
      </div>
    `;

    if (data.status === 'success' && data.low_estimate) {
      html += `<div class="valuation-price">${data.estimated_value}</div>`;
      html += `<div class="valuation-grid">`;

      if (data.low_estimate) {
        html += `
          <div class="valuation-item">
            <div class="valuation-label">Low Estimate</div>
            <div class="valuation-value">$${data.low_estimate}</div>
          </div>
        `;
      }

      if (data.mid_estimate) {
        html += `
          <div class="valuation-item">
            <div class="valuation-label">Average</div>
            <div class="valuation-value">$${data.mid_estimate}</div>
          </div>
        `;
      }

      if (data.high_estimate) {
        html += `
          <div class="valuation-item">
            <div class="valuation-label">High Estimate</div>
            <div class="valuation-value">$${data.high_estimate}</div>
          </div>
        `;
      }

      if (data.recent_sales_count) {
        html += `
          <div class="valuation-item">
            <div class="valuation-label">Recent Sales</div>
            <div class="valuation-value">${data.recent_sales_count}</div>
          </div>
        `;
      }

      html += `</div>`;

      if (data.confidence) {
        html += `<div class="valuation-confidence ${data.confidence}">${data.confidence.toUpperCase()} confidence</div>`;
      }

      if (data.data_sources && data.data_sources.length > 0) {
        html += `<div class="valuation-sources">📊 Data from: ${data.data_sources.join(', ')}</div>`;
      }
    } else if (data.status === 'no_data') {
      html += `<div class="valuation-loading">Enable API keys (eBay, Card Ladder) in Fly.io settings to show market valuations.</div>`;
    } else {
      html += `<div class="valuation-error">Error: ${data.estimated_value || 'Could not fetch valuation'}</div>`;
    }

    valuationDiv.innerHTML = html;
  } catch (error) {
    valuationDiv.innerHTML = `
      <div class="valuation-header">
        <span class="valuation-icon">💰</span>
        <h3 class="valuation-title">Market Valuation</h3>
      </div>
      <div class="valuation-error">Valuation service not available yet. API keys needed.</div>
    `;
  }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  injectValuationStyles();
});

// Export for use in main script
window.fetchAndDisplayValuation = fetchAndDisplayValuation;
