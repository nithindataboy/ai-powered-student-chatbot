// Cache DOM elements
const elements = {
    searchInput: document.getElementById('searchInput'),
    searchBtn: document.getElementById('searchBtn'),
    aiSearchBtn: document.getElementById('aiSearchBtn'),
    faqSelect: document.getElementById('faqSelect'),
    responseContainer: document.getElementById('responseContainer'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    searchSuggestions: document.querySelector('.search-suggestions'),
    searchHistory: document.querySelector('.search-history')
};

// Initialize speech recognition and synthesis
let recognition = null;
let synthesis = null;
let isListening = false;
let isSpeaking = false;

// Initialize speech recognition
function initSpeechRecognition() {
    try {
        recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';
        
        recognition.onresult = (event) => {
            const text = event.results[0][0].transcript;
            if (elements.searchInput) {
                elements.searchInput.value = text;
                handleSearch();
            }
        };
        
        recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            stopListening();
        };
        
        recognition.onend = () => {
            stopListening();
        };
    } catch (error) {
        console.error('Speech recognition not supported:', error);
    }
}

// Initialize speech synthesis
function initSpeechSynthesis() {
    try {
        synthesis = window.speechSynthesis;
    } catch (error) {
        console.error('Speech synthesis not supported:', error);
    }
}

// Initialize speech features
initSpeechRecognition();
initSpeechSynthesis();

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    // Load FAQ questions
    loadFAQQuestions();
    
    // Load search history
    loadSearchHistory();
    
    // Add event listeners
    if (elements.searchBtn) {
        elements.searchBtn.addEventListener('click', handleSearch);
    }
    if (elements.aiSearchBtn) {
        elements.aiSearchBtn.addEventListener('click', handleAISearch);
    }
    if (elements.faqSelect) {
        elements.faqSelect.addEventListener('change', handleFAQSelect);
    }
    if (elements.searchInput) {
        elements.searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                handleSearch();
            }
        });
    }
});

// Load FAQ questions
async function loadFAQQuestions() {
    try {
        const response = await fetch('/api/faq');
        if (!response.ok) throw new Error('Failed to load FAQ questions');
        
        const data = await response.json();
        if (data.error) throw new Error(data.error);
        
        // Clear existing options
        elements.faqSelect.innerHTML = '<option value="">Select a question...</option>';
        
        // Add new options
        data.questions.forEach(question => {
            const option = document.createElement('option');
            option.value = question;
            option.textContent = question;
            elements.faqSelect.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading FAQ questions:', error);
        showError('Failed to load FAQ questions');
    }
}

// Handle Google search
async function handleSearch() {
    const query = elements.searchInput.value.trim();
    
    if (!query) {
        showError('Please enter a search query');
        return;
    }
    
    try {
        // Show loading state
        showLoading();
        
        // Make API call
        const response = await fetch('/api/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query })
        });
        
        if (!response.ok) {
            throw new Error('Search failed');
        }
        
        const data = await response.json();
        
        // Display results
        displayResults(data);
        
        // Add to search history
        addToSearchHistory(query);
        
    } catch (error) {
        console.error('Search error:', error);
        showError('Failed to perform search. Please try again.');
    } finally {
        hideLoading();
    }
}

// Handle BERT search
async function handleAISearch() {
    const query = elements.searchInput.value.trim();
    if (!query) {
        showError('Please enter a search query');
        return;
    }

    showLoading();
    try {
        const response = await fetch('/api/faq/answer', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question: query })
        });

        if (!response.ok) {
            throw new Error('Search failed');
        }

        const data = await response.json();
        displayFAQAnswer(data);
        
        addToSearchHistory(query);
        
    } catch (error) {
        console.error('BERT search error:', error);
        showError('Failed to perform BERT search. Please try again.');
    } finally {
        hideLoading();
    }
}

// Handle FAQ selection
async function handleFAQSelect(event) {
    const question = event.target.value;
    if (!question) return;

    showLoading();
    try {
        const response = await fetch('/api/faq/answer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
        });

        if (!response.ok) {
            throw new Error('FAQ search failed');
        }

        const data = await response.json();
        displayFAQAnswer(data);
        
    } catch (error) {
        console.error('FAQ error:', error);
        showError('Failed to get FAQ answer. Please try again.');
    } finally {
        hideLoading();
    }
}

// Display search results
function displayResults(data) {
    const responseContainer = document.getElementById('responseContainer');
    if (!responseContainer) {
        console.error('Response container not found');
        return;
    }
    
    // Clear previous results
    responseContainer.innerHTML = '';
    
    // Check if we have results
    if (!data.results || data.results.length === 0) {
        responseContainer.innerHTML = '<div class="no-results">No results found</div>';
        return;
    }
    
    // Display FAQ answer if available
    if (data.faq_answer) {
        const faqSection = document.createElement('div');
        faqSection.className = 'faq-section';
        faqSection.innerHTML = `
            <h3>Quick Answer</h3>
            <div class="faq-card">
                <h4>${data.faq_answer.question}</h4>
                <p>${data.faq_answer.answer}</p>
                <small>Source: ${data.faq_answer.source}</small>
            </div>
        `;
        responseContainer.appendChild(faqSection);
    }
    
    // Separate text and image results
    const textResults = data.results.filter(r => r.type === 'text');
    const imageResults = data.results.filter(r => r.type === 'image');
    
    // Display text results
    if (textResults.length > 0) {
        const textSection = document.createElement('div');
        textSection.className = 'text-results';
        textSection.innerHTML = '<h3>Search Results</h3>';
        
        textResults.forEach(result => {
            const resultCard = document.createElement('div');
            resultCard.className = 'result-card text-card';
            resultCard.innerHTML = `
                <h4>${result.title || 'No title'}</h4>
                <p>${result.snippet || 'No description available'}</p>
                <a href="${result.link}" target="_blank" class="read-more">Read more</a>
            `;
            textSection.appendChild(resultCard);
        });
        
        responseContainer.appendChild(textSection);
    }
    
    // Display image results
    if (imageResults.length > 0) {
        const imageSection = document.createElement('div');
        imageSection.className = 'image-results';
        imageSection.innerHTML = '<h3>Related Images</h3>';
        
        imageResults.forEach(result => {
            const resultCard = document.createElement('div');
            resultCard.className = 'result-card image-card';
            resultCard.innerHTML = `
                <img src="${result.image_url}" alt="${result.title || 'Image'}" onerror="this.src='static/images/placeholder.png'">
                <h4>${result.title || 'No title'}</h4>
                <p>${result.snippet || 'No description available'}</p>
                <a href="${result.link}" target="_blank" class="read-more">View source</a>
            `;
            imageSection.appendChild(resultCard);
        });
        
        responseContainer.appendChild(imageSection);
    }
}

function displayFAQAnswer(data) {
    const responseContainer = document.getElementById('responseContainer');
    if (!responseContainer) {
        console.error('Response container not found');
        return;
    }

    if (!data.answer) {
        showError('No answer found for this FAQ question.');
        return;
    }

    responseContainer.innerHTML = `
        <div class="faq-section">
            <h3>FAQ Answer</h3>
            <div class="faq-card">
                <h4>${data.question}</h4>
                <p>${data.answer}</p>
                ${data.confidence ? `<small>Confidence: ${data.confidence}%</small>` : ''}
                ${data.source ? `<small>Source: ${data.source}</small>` : ''}
            </div>
        </div>`;
}

// Helper functions
function showLoading() {
    elements.loadingOverlay.style.display = 'flex';
}

function hideLoading() {
    elements.loadingOverlay.style.display = 'none';
}

function showError(message) {
    const responseContainer = document.getElementById('responseContainer');
    if (!responseContainer) {
        console.error('Response container not found');
        return;
    }

    responseContainer.innerHTML = `
        <div class="error-message">
            <i class="fas fa-exclamation-circle"></i>
            <p>${escapeHtml(message)}</p>
            <button onclick="handleSearch()" class="retry-button">
                <i class="fas fa-redo"></i> Retry
            </button>
        </div>
    `;
}

// Search history functions
function loadSearchHistory() {
    try {
        const history = JSON.parse(localStorage.getItem('searchHistory') || '[]');
        if (history.length > 0) {
            elements.searchHistory.innerHTML = `
                <div class="history-header">
                    <h4>Recent Searches</h4>
                    <button onclick="clearSearchHistory()" class="clear-history">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
                <ul>
                    ${history.map(item => `
                        <li>
                            <button onclick="useSearchHistory('${escapeHtml(item)}')">
                                <i class="fas fa-history"></i>
                                ${escapeHtml(item)}
                            </button>
                        </li>
                    `).join('')}
                </ul>
            `;
        }
    } catch (error) {
        console.error('Error loading search history:', error);
    }
}

function addToSearchHistory(query) {
    try {
        let history = JSON.parse(localStorage.getItem('searchHistory') || '[]');
        history = [query, ...history.filter(item => item !== query)].slice(0, 5);
        localStorage.setItem('searchHistory', JSON.stringify(history));
        loadSearchHistory();
    } catch (error) {
        console.error('Error adding to search history:', error);
    }
}

function clearSearchHistory() {
    try {
        localStorage.removeItem('searchHistory');
        elements.searchHistory.innerHTML = '';
    } catch (error) {
        console.error('Error clearing search history:', error);
    }
}

function useSearchHistory(query) {
    elements.searchInput.value = query;
    handleSearch();
}

function escapeHtml(text) {
    if (!text) return '';
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

// Add CSS for better image display
const style = document.createElement('style');
style.textContent = `
    .search-results-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }

    .results-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin-top: 1rem;
    }

    .result-card {
        height: 100%;
    }

    .result-card .card {
        height: 100%;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        display: flex;
        flex-direction: column;
    }

    .image-container {
        position: relative;
        padding-top: 75%; /* 4:3 Aspect Ratio */
        background: #f8f9fa;
        overflow: hidden;
    }

    .image-container img {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        object-fit: contain;
    }

    .result-card .card-body {
        padding: 1.5rem;
        flex-grow: 1;
    }

    .result-card .card-title {
        color: #1a237e;
        margin-bottom: 1rem;
        font-size: 1.1rem;
        font-weight: 600;
    }

    .result-card .card-text {
        color: #666;
        line-height: 1.6;
        margin-bottom: 1rem;
    }

    .btn-primary {
        background-color: #1a237e;
        border-color: #1a237e;
    }

    .btn-primary:hover {
        background-color: #0d47a1;
        border-color: #0d47a1;
    }
`;
document.head.appendChild(style);