const searchBtn = document.getElementById('searchBtn');
const topicInput = document.getElementById('topicInput');
const result = document.getElementById('result');
const chatSection = document.getElementById('chatSection');
const chatHistory = document.getElementById('chatHistory');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');

const sessionId = 'session_' + Date.now();

function addMessage(text, sender) {
    const message = document.createElement('div');
    message.classList.add('message', sender);
    if (sender === 'ai') {
    message.innerHTML = marked.parse(text);
} else {
    message.textContent = text;
}
    chatHistory.appendChild(message);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

searchBtn.addEventListener('click', async () => {
    const topic = topicInput.value.trim();

    if (!topic) {
        alert('Please enter a research topic.');
        return;
    }

    searchBtn.textContent = 'Researching...';
    searchBtn.disabled = true;
    result.style.display = 'none';

    const response = await fetch('http://127.0.0.1:8000/research', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({topic: topic, session_id: sessionId})
    });

    const data = await response.json();
    result.innerHTML = marked.parse(data.report);
    result.style.display = 'block';
    chatSection.style.display = 'block';

    searchBtn.textContent = 'Research';
    searchBtn.disabled = false;
});

sendBtn.addEventListener('click', async () => {
    const message = chatInput.value.trim();

    if (!message) {
        alert('Please enter a question.');
        return;
    }

    addMessage(message, 'user');
    chatInput.value = '';
    sendBtn.textContent = 'Sending...';
    sendBtn.disabled = true;

    const response = await fetch('http://127.0.0.1:8000/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({message: message, session_id: sessionId})
    });

    const data = await response.json();
    addMessage(data.response, 'ai');

    sendBtn.textContent = 'Send';
    sendBtn.disabled = false;
});