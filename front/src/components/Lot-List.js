import '../App.css';
import { useState, useEffect } from 'react';
import axios from 'axios';
import React from 'react';

function LotList() {
  const [lotInfo, setLotInfo] = useState([]);
  const [currentPage, setCurrentPage] = useState(0);

  // Number of items to display per page
  const itemsPerPage = 10;

  useEffect(() => {
    const loadLots = async () => {
      const response = await axios.get('/api/lots');
      setLotInfo(response.data);
    };
    loadLots();
  }, []);

  // Calculate start/end index for slicing
  const startIndex = currentPage * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;

  // Slice the data for the current page
  const currentItems = lotInfo.slice(startIndex, endIndex);

  // Handlers for navigating pages
  const handleNext = () => {
    if (endIndex < lotInfo.length) {
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
            <th scope="col">Lot ID</th>
            <th scope="col">Date</th>
            <th scope="col">Supplier</th>
            <th scope="col">Quantity</th>
            <th scope="col">Grade</th>
          </tr>
        </thead>
        <tbody>
          {currentItems.map((lot) => (
            <tr key={lot.LotId}>
              <td>{lot.LotId}</td>
              <td>{lot.Date}/{lot.Month}</td>
              <td>{lot.SuppName}</td>
              <td>{(lot.AccQty ?? 0).toFixed(2)} kg</td>

              <td>{lot.Grade.toUpperCase()}</td>
             
            </tr>
          ))}
        </tbody>
      </table>

      <div>
        <center>{/* Only show Previous button if not on the first page */}
        {currentPage > 0 && (
          <button class="btn btn-success btn-sm" onClick={handlePrevious}>
            Previous
          </button>
        )}
       &emsp; 
        {/* Only show Next button if there are more items to show */}
        {endIndex < lotInfo.length && (
          <button class="btn btn-success btn-sm" onClick={handleNext}>
            Next >
          </button>
        )} </center>
      </div>
    </div>
  );
}

export default LotList;
