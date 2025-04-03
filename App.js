const App = () => {
    const [responses, setResponses] = React.useState([]);
    const [isLoading, setIsLoading] = React.useState(false);
    const [isRecording, setIsRecording] = React.useState(false);
    const [questions, setQuestions] = React.useState([]);
    const [recognition, setRecognition] = React.useState(null);

    React.useEffect(() => {
        // Initialize speech recognition
        if ('webkitSpeechRecognition' in window) {
            const recognition = new window.webkitSpeechRecognition();
            recognition.continuous = true;
            recognition.interimResults = true;
            recognition.lang = 'en-US';

            recognition.onresult = (event) => {
                const transcript = Array.from(event.results)
                    .map(result => result[0].transcript)
                    .join('');
                document.getElementById('userInput').value = transcript;
            };

            recognition.onerror = (event) => {
                console.error('Speech recognition error:', event.error);
                setIsRecording(false);
            };

            setRecognition(recognition);
        }
    }, []);

    const handleSearch = async (query) => {
        setIsLoading(true);
        try {
            const response = await fetch('/api/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query }),
            });

            if (!response.ok) throw new Error('Search failed');
            const data = await response.json();
            setResponses(data.responses);
        } catch (error) {
            console.error('Search error:', error);
            setResponses([{
                type: 'text',
                content: 'Sorry, there was an error processing your request.',
                source: 'System'
            }]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleGoogleSearch = () => {
        const query = document.getElementById('userInput').value;
        if (query.trim()) {
            window.open(`https://www.google.com/search?q=${encodeURIComponent(query)}`, '_blank');
        }
    };

    const handleVoiceInput = () => {
        if (!recognition) {
            alert('Speech recognition is not supported in your browser.');
            return;
        }

        if (isRecording) {
            recognition.stop();
            setIsRecording(false);
        } else {
            recognition.start();
            setIsRecording(true);
        }
    };

    const handleFAQSelect = async (question) => {
        setIsLoading(true);
        try {
            const response = await fetch('/api/faq/answer', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question }),
            });

            if (!response.ok) throw new Error('Failed to get FAQ answer');
            const data = await response.json();
            setResponses([{
                type: 'text',
                content: data.answer,
                source: 'FAQ'
            }]);
        } catch (error) {
            console.error('FAQ error:', error);
            setResponses([{
                type: 'text',
                content: 'Sorry, there was an error retrieving the FAQ answer.',
                source: 'System'
            }]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="app-container">
            <Header />
            <main className="main-content">
                <Search
                    onSearch={handleSearch}
                    onGoogleSearch={handleGoogleSearch}
                    onVoiceInput={handleVoiceInput}
                    isRecording={isRecording}
                />
                <FAQ
                    questions={questions}
                    onQuestionSelect={handleFAQSelect}
                />
                <Response
                    responses={responses}
                    isLoading={isLoading}
                />
            </main>
        </div>
    );
}; 