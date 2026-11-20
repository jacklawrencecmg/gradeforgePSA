// Hook valuation display into form submission
document.addEventListener('DOMContentLoaded', () => {
  const analyzeForm = document.getElementById('form');
  if (!analyzeForm) return;

  // Observe result div for changes
  const resultDiv = document.getElementById('result');
  if (!resultDiv) return;

  const observer = new MutationObserver((mutations) => {
    // Check if result contains analysis data
    const resultText = resultDiv.textContent;
    if (resultText && resultText.includes('id')) {
      // Extract analysis ID from the result div
      const idMatch = resultDiv.innerHTML.match(/"id":"([A-Za-z0-9]+)"/);
      const analysisId = idMatch ? idMatch[1] : null;

      if (analysisId && analysisId.length > 10) {
        // Check if valuation already displayed
        if (!document.getElementById(`valuation-${analysisId}`)) {
          // Fetch and display valuation
          if (window.fetchAndDisplayValuation) {
            window.fetchAndDisplayValuation(analysisId);
          }
        }
      }
    }
  });

  observer.observe(resultDiv, { childList: true, subtree: true });
});
