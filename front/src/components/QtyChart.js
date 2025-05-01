import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';

import { useState,useEffect } from 'react';
import axios from 'axios';







function QtyChart() {

const [sampleData, setSampleData] = useState([]);


useEffect(()=>{

  const loadData = async()=>{


    const response = await axios.get(`/api/qty/chart`);

    const sampleData = response.data;

    setSampleData(sampleData);

  };

  loadData();


},[]);




  return (
    <div style={{ width: '100%', height: 400 }}>
      <center><h4>Last 7 Days</h4></center>
      <ResponsiveContainer>
        <BarChart data={sampleData}  margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="Date" />
          <YAxis />
          <Tooltip
  formatter={(value) => Number(value).toFixed(2)}
/>
          

          {/* These Bars will correspond to the keys in your data objects */}
          <Bar dataKey="Quantity" fill="#069b49" />
          <Bar dataKey="pv" fill="#069b49" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export default QtyChart;
