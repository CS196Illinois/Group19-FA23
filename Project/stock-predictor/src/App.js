import React, { useState } from 'react';
import './App.css';

function App() {
  const [ticker, setTicker] = useState('');
  const [prediction, setPrediction] = useState('');

  const handlePredictClick = async () => {
    const response = await fetch('http://127.0.0.1:5000/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ ticker: ticker })
    });
    const data = await response.json();
    setPrediction(data.prediction);
  };

  return (
    <div className="App">
      <input
        value={ticker}
        onChange={e => setTicker(e.target.value)}
        placeholder="Enter stock ticker"
      />
      <button onClick={handlePredictClick}>Predict</button>
      <p>{prediction}</p>
    </div>
  );
}

export default App;
