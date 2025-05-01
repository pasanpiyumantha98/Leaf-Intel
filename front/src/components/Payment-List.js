import '../App.css';
import { useState, useEffect } from 'react';
import axios from 'axios';
import React from 'react';

function PaymentList() {
  const [payInfo, setPayInfo] = useState([]);
  const [currentPage, setCurrentPage] = useState(0);

  // How many items to display per page
  const itemsPerPage = 10;

  useEffect(() => {
    const loadPayments = async () => {
      const response = await axios.get('/api/payments');
      setPayInfo(response.data);
    };

    loadPayments();
  }, []);

  // Calculate the start/end indexes for the current page
  const startIndex = currentPage * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;

  // Slice the original array to get items for the current page
  const currentItems = payInfo.slice(startIndex, endIndex);

  // Handlers for next/previous pages
  const handleNext = () => {
    if (endIndex < payInfo.length) {
      setCurrentPage(currentPage + 1);
    }
  };

  const handlePrevious = () => {
    if (currentPage > 0) {
      setCurrentPage(currentPage - 1);
    }
  };

  return (
    <div>
      <table className="table">
        <thead>
          <tr>
            <th scope="col">ID</th>
            <th scope="col">Date</th>
            <th scope="col">Type</th>
            <th scope="col">Supplier ID</th>
            <th scope="col">Amount</th>
            <th scope="col">Issued by</th>
          </tr>
        </thead>
        <tbody>
          {currentItems.map((item) => (
            <tr key={item.TransId}>
              <td scope="row">{item.TransId}</td>
              <td>{item.Year}/{item.Month}/{item.Date}</td>
              <td>{item.Type}</td>
              <td>{item.SuppId}</td>
              <td>{item.Amount}</td>
              <td>{item.Staff}</td>
             
            </tr>
          ))}
        </tbody>
      </table>

      <div>
        {currentPage > 0 && (
          <button class="btn btn-success btn-sm" onClick={handlePrevious}> Previous </button>
        )}
        &emsp; 
        {endIndex < payInfo.length && (
          <button class="btn btn-success btn-sm" onClick={handleNext}> Next > </button>
        )}
      </div>
    </div>
  );
}

export default PaymentList;
