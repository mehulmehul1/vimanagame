import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './styles/editor.css';

// Initialize the editor application
const rootElement = document.getElementById('root');

if (!rootElement) {
    throw new Error('Root element not found');
}

const root = ReactDOM.createRoot(rootElement);

root.render(
    <React.StrictMode>
        <App />
    </React.StrictMode>
);

// Hot module replacement for development
if (import.meta.hot) {
    import.meta.hot.accept('./App', () => {
        root.render(
            <React.StrictMode>
                <App />
            </React.StrictMode>
        );
    });
}
