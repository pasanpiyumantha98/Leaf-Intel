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



function BillingChart() {

const [sampleData, setSampleData] = useState([]);


useEffect(()=>{

  const loadData = async()=>{


    const response = await axios.get(`/api/billing/chart`);

    const sampleData = response.data;

    setSampleData(sampleData);

  };

  loadData();


},[]);




  return (
    <div style={{ width: '100%', height: 400 }}>
      <h4>Last 6 Months</h4>
      <ResponsiveContainer>
        <BarChart data={sampleData}  margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="month" />
          <YAxis />
          <Tooltip
  formatter={(value) => Number(value).toFixed(2)}
/>
          

          {/* These Bars will correspond to the keys in your data objects */}
          <Bar dataKey="Amount" fill="#069b49" />
          <Bar dataKey="pv" fill="#069b49" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export default BillingChart;
