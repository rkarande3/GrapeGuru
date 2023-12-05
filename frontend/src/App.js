import logo from './logo.svg';
import './App.css';
import React, { useState } from 'react';
import axios from 'axios';
import './App.css';
import { ChakraProvider } from '@chakra-ui/react'

//data will be the string we send from our server
const apiCall = () => {
  axios.get('http://localhost:8080').then((data) => {
    //this console.log will be in our frontend console
    console.log(data)
  })
}


function App() {
  const [inputData, setInputData] = useState('');
  const [result, setResult] = useState('');

  const handleSubmit = async (event) => {
    event.preventDefault();

    try {
      setResult("");
      console.log('Submitting data:', inputData);
      const response = await axios.post('http://localhost:8080/api/process_data', {
        data: inputData,
      });
      console.log(response)
      setResult(response.data.message);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
      <div className="App" style={{ height: '100vh', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' , padding: '20px'}}>
        <h1 className="title">GrapeGuru </h1>
            <label className="input_desc" htmlFor="wine_des">Input Wine Description :</label>
            <div>
            <textarea
              style={{ width: '300px', height: '100px', whiteSpace: 'normal', wordWrap: 'break-word'}}
              type="text"
              value={inputData}
              onChange={(e) => setInputData(e.target.value)}
            />
            <br />
            <button style={{backgroundColor: '#e3b5c6', color: 'black', fontWeight: 'bold', borderRadius: '10px', padding: '3px 5px', outline: '2px solid white'}} className="button_format" type="button" onClick={handleSubmit}>
              Submit
            </button>
            </div>
        <p style={{color:'#e3b5c6', fontWeight: 'bold'}}> Result: {result}</p>
      </div>
  );
}

export default App;

